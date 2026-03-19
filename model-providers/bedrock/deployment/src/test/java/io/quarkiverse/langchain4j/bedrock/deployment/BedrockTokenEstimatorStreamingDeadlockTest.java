package io.quarkiverse.langchain4j.bedrock.deployment;

import static com.github.tomakehurst.wiremock.client.WireMock.aResponse;
import static com.github.tomakehurst.wiremock.client.WireMock.post;
import static com.github.tomakehurst.wiremock.client.WireMock.stubFor;
import static com.github.tomakehurst.wiremock.client.WireMock.urlPathEqualTo;
import static org.assertj.core.api.Assertions.assertThat;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import jakarta.inject.Inject;

import org.jboss.shrinkwrap.api.ShrinkWrap;
import org.jboss.shrinkwrap.api.spec.JavaArchive;
import org.junit.jupiter.api.DisplayNameGeneration;
import org.junit.jupiter.api.DisplayNameGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.UserMessage;
import io.quarkiverse.langchain4j.RegisterAiService;
import io.quarkiverse.langchain4j.bedrock.runtime.jaxrsclient.JaxRsSdkHttpClient;
import io.quarkus.arc.Arc;
import io.quarkus.test.QuarkusUnitTest;
import io.smallrye.common.vertx.VertxContext;
import io.smallrye.mutiny.Multi;
import io.vertx.core.Context;
import io.vertx.core.Vertx;
import software.amazon.awssdk.http.ContentStreamProvider;
import software.amazon.awssdk.http.HttpExecuteRequest;
import software.amazon.awssdk.http.SdkHttpClient;
import software.amazon.awssdk.http.SdkHttpMethod;
import software.amazon.awssdk.http.SdkHttpRequest;

/**
 * Reproducer: {@link TokenCountEstimator} making HTTP calls via {@link JaxRsSdkHttpClient} deadlocks
 * during streaming response handling.
 *
 * <p>
 * When streaming completes, {@code QuarkusAiServiceStreamingResponseHandler} dispatches
 * {@code addToMemory()} via {@code executeBlocking(ordered=true)} on a Vert.x worker thread.
 * If the estimator makes a sync HTTP call through resteasy-reactive, the response dispatch
 * needs the same Vert.x context — blocked by the current task. Deadlock.
 *
 * <p>
 * The test must run from a Vert.x duplicated context (like a real HTTP endpoint) because
 * {@code switchToWorkerForEmission} is only true when the caller is on a Vert.x worker thread.
 */
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class BedrockTokenEstimatorStreamingDeadlockTest extends BedrockTestBase {

    @RegisterExtension
    static final QuarkusUnitTest unitTest = new QuarkusUnitTest()
            .setArchiveProducer(() -> ShrinkWrap.create(JavaArchive.class)
                    .addClass(TestCredentialsProvider.class)
                    .addClass(BedrockStreamHelper.class))
            .overrideRuntimeConfigKey("quarkus.langchain4j.bedrock.chat-model.model-id", "amazon.titan-text-express-v1")
            .overrideRuntimeConfigKey("quarkus.langchain4j.bedrock.chat-model.aws.region", "eu-central-1")
            .overrideRuntimeConfigKey("quarkus.langchain4j.bedrock.chat-model.aws.endpoint-override",
                    "http://localhost:%d".formatted(WM_PORT))
            .overrideRuntimeConfigKey("quarkus.langchain4j.bedrock.chat-model.aws.credentials-provider",
                    "TestCredentialsProvider");

    @Inject
    MyAiService service;

    @Inject
    Vertx vertx;

    @Test
    void streaming_with_token_estimator_should_not_deadlock() throws InterruptedException {
        stubFor(post(urlPathEqualTo("/count-tokens"))
                .willReturn(aResponse()
                        .withStatus(200)
                        .withHeader("Content-Type", "application/json")
                        .withBody("{\"inputTokens\": 42}")));

        Context context = VertxContext.getOrCreateDuplicatedContext(vertx);
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<List<String>> resultRef = new AtomicReference<>();
        AtomicReference<Throwable> errorRef = new AtomicReference<>();

        context.executeBlocking(v -> {
            try {
                Arc.container().requestContext().activate();
                List<String> list = service.chat("123", "Say hello")
                        .collect().asList()
                        .ifNoItem().after(Duration.ofSeconds(10)).fail()
                        .await().atMost(Duration.ofSeconds(15));
                resultRef.set(list);
            } catch (Throwable t) {
                errorRef.set(t);
            } finally {
                Arc.container().requestContext().deactivate();
                latch.countDown();
            }
        }, false);

        assertThat(latch.await(20, TimeUnit.SECONDS))
                .as("Test should complete within 20s (deadlock if it doesn't)")
                .isTrue();

        if (errorRef.get() != null) {
            throw new AssertionError("Streaming with TokenCountEstimator deadlocked", errorRef.get());
        }

        assertThat(resultRef.get()).containsExactly("Hello!");
    }

    // -- Fakes --

    public static class FakeStreamingChatModel implements StreamingChatModel {
        @Override
        public void doChat(ChatRequest chatRequest, StreamingChatResponseHandler handler) {
            Vertx vertx = Arc.container().select(Vertx.class).get();
            Context ctxt = vertx.getOrCreateContext();
            ctxt.runOnContext(v -> {
                handler.onPartialResponse("Hello!");
                ctxt.runOnContext(v2 -> handler.onCompleteResponse(
                        ChatResponse.builder().aiMessage(new AiMessage("Hello!")).build()));
            });
        }
    }

    public static class FakeStreamingChatModelSupplier implements Supplier<StreamingChatModel> {
        @Override
        public StreamingChatModel get() {
            return new FakeStreamingChatModel();
        }
    }

    public static class HttpCallingTokenCountEstimator implements TokenCountEstimator {

        private final SdkHttpClient httpClient = JaxRsSdkHttpClient.builder()
                .connectionTimeout(Duration.ofSeconds(5))
                .readTimeout(Duration.ofSeconds(5))
                .build();

        @Override
        public int estimateTokenCountInText(String text) {
            return callCountTokensApi();
        }

        @Override
        public int estimateTokenCountInMessage(ChatMessage message) {
            return callCountTokensApi();
        }

        @Override
        public int estimateTokenCountInMessages(Iterable<ChatMessage> messages) {
            return callCountTokensApi();
        }

        private int callCountTokensApi() {
            try {
                SdkHttpRequest request = SdkHttpRequest.builder()
                        .uri(URI.create("http://localhost:%d/count-tokens".formatted(WM_PORT)))
                        .method(SdkHttpMethod.POST)
                        .putHeader("content-type", "application/json")
                        .build();

                HttpExecuteRequest executeRequest = HttpExecuteRequest.builder()
                        .request(request)
                        .contentStreamProvider(ContentStreamProvider.fromUtf8String("{}"))
                        .build();

                httpClient.prepareRequest(executeRequest).call();
                return 42;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class TokenWindowMemoryProviderSupplier implements Supplier<ChatMemoryProvider> {
        @Override
        public ChatMemoryProvider get() {
            return memoryId -> TokenWindowChatMemory.withMaxTokens(1000, new HttpCallingTokenCountEstimator());
        }
    }

    @RegisterAiService(streamingChatLanguageModelSupplier = FakeStreamingChatModelSupplier.class, chatMemoryProviderSupplier = TokenWindowMemoryProviderSupplier.class)
    public interface MyAiService {
        Multi<String> chat(@MemoryId String id, @UserMessage String query);
    }
}
