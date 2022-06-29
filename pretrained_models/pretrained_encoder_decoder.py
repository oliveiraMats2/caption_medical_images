from transformers import ConvNextModel, ResNetModel, T5ForConditionalGeneration
import torch 

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class ConvNext2T5Model(nn.Module):
    def __init__(self, pretrained_encoder="convnext_tiny"):
        super().__init__()
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-small")
        print(get_n_params(self.decoder))
        #self.decoder.lm_head = nn.Identity()
        if pretrained_encoder is "convnext_tiny":
            self.encoder = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
            in_channels=768
        elif pretrained_encoder is "resnet18":
            self.encoder = ResNetModel.from_pretrained("microsoft/resnet-18")
            in_channels = 512


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

        #print(out.shape)
        out = self.encoder.forward(out, return_dict = False)[0]
        #print(out.shape)
        out = self.connect_enc_dec(out)
        #print(out.shape)
        out = out.permute(0, 2, 3, 1).reshape(-1, 81, 512)
        #print(out.shape)

        if len(args)>1:
            labels = args[1]
            out = self.decoder(inputs_embeds=out, labels=labels)
        else:
            out = self.decoder.generate(inputs_embeds=out)
        #print(out[0])
        return out

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ConvNext2T5Model(pretrained_encoder="convnext_tiny")
    model = model.to(device)
    model.n_params()
