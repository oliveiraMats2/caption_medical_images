from transformers import ConvNextModel, ResNetModel, T5ForConditionalGeneration, T5Tokenizer
import torch
from torch import nn
from tqdm import tqdm



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



class ConvNext2T5Model(nn.Module):
    def __init__(self, tokenizer=False, pretrained_encoder="convnext-tiny", pretrained_decoder="t5-small", max_phrase_length=256, min_phrase_length=5):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_phrase_length = max_phrase_length
        self.min_phrase_length = min_phrase_length

        if pretrained_encoder == "convnext-tiny":
            self.encoder = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
            in_channels=768
        elif pretrained_encoder == "convnext-base":
            self.encoder = ConvNextModel.from_pretrained("facebook/convnext-base-224")
            in_channels=1024
        elif pretrained_encoder == "convnext-large":
            self.encoder = ConvNextModel.from_pretrained("facebook/convnext-large-224")
            in_channels=1536
        elif pretrained_encoder == "resnet18":
            self.encoder = ResNetModel.from_pretrained("microsoft/resnet-18")
            in_channels = 512

        
        if pretrained_decoder == "t5-small":
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-small")
        elif pretrained_decoder == "t5-base":
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-base")
        elif pretrained_decoder == "t5-large":
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-large")

        print(in_channels)



        self.connect_enc_dec = nn.Conv2d(in_channels=in_channels, 
                                         out_channels=self.decoder.config.d_model,
                                         kernel_size=1,
                                         stride=1,
                                         padding=1)
        
    def n_params(self):
        encoder_params = get_n_params(self.encoder)
        decoder_params = get_n_params(self.decoder)
        connector_params = get_n_params(self.connect_enc_dec)
        total_params = get_n_params(self)
        print(f"""
                Encoder params: {encoder_params}
                Decoder params: {decoder_params}
                Connector params: {connector_params}
                Total params:  {total_params}
                """
                )
        
    def forward(self, *args):
        out = args[0]

        out = self.encoder.forward(out, return_dict = False)[0]
        out = self.connect_enc_dec(out)
        out = out.permute(0, 2, 3, 1).reshape(-1, 81, self.decoder.config.d_model)

        if len(args)>1:
            labels = args[1]
            out = self.decoder(inputs_embeds=out, labels=labels, return_dict=True)
        else:
            out = self.decoder.generate(inputs_embeds=out,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=self.min_phrase_length,
                                    max_length=self.max_phrase_length,
                                    early_stopping=True)
        return out



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = ConvNext2T5Model(pretrained_encoder="convnext-tiny", tokenizer=tokenizer)
    model = model.to(device)
    model.n_params()
