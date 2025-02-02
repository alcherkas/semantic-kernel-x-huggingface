using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.TextGeneration;
#pragma warning disable SKEXP0010

var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("llama-3.2-1b-instruct", new Uri("http://127.0.0.1:1234/v1"), "API_KEY")
    .Build();

var textGeneration = kernel.GetRequiredService<ITextGenerationService>();
var result = await textGeneration.GetTextContentsAsync("Summarize: Paris is the capital and most populous city of France, with\n          an estimated population of 2,175,601 residents as of 2018,\n          in an area of more than 105 square kilometres (41 square\n          miles). The City of Paris is the centre and seat of\n          government of the region and province of Île-de-France, or\n          Paris Region, which has an estimated population of\n          12,174,880, or about 18 percent of the population of France\n          as of 2017.");

Console.WriteLine(result[0].Text);