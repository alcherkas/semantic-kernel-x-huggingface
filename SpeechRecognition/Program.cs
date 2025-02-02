using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AudioToText;

#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010

var client = new HttpClient { BaseAddress = new Uri("http://127.0.0.1:5001/v1") };
var kernel = Kernel.CreateBuilder()
    .AddOpenAIAudioToText("openai/whisper-large-v3", "key", httpClient: client)
    .Build();

var audioData = File.ReadAllBytes("sample-0.wav");
var audioToTextService = kernel.GetRequiredService<IAudioToTextService>();
var result = await audioToTextService.GetTextContentsAsync(new AudioContent(audioData, "audio/wav"));

Console.WriteLine(result[0].Text);
