using System.Numerics.Tensors;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010

var client = new HttpClient { BaseAddress = new Uri("http://localhost:5001") };
var kernel = Kernel.CreateBuilder()
    .AddOpenAITextEmbeddingGeneration("nomic-ai/modernbert-embed-base", "key", httpClient: client)
    .Build();

List<string> sentences1 = ["The cat sits outside", "A man is playing guitar?", "The movies are awesome"];
var embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
var embeddings1 = await embeddingService.GenerateEmbeddingsAsync(sentences1);

string[] sentences2 = ["The dog plays in the garden", "A woman watches TV", "The new movie is so great"];
var embeddings2 = await embeddingService.GenerateEmbeddingsAsync(sentences2);
for (var i = 0; i < sentences2.Length; i++)
{
    var similarity = TensorPrimitives.CosineSimilarity(embeddings1[i].Span, embeddings2[i].Span);
    Console.WriteLine($"Text1: {sentences1[i]}");
    Console.WriteLine($"Text2: {sentences2[i]}");
    Console.WriteLine($"Score: {similarity}");
}
