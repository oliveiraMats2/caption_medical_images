from transformers import ConvNextModel, ResNetModel, T5ForConditionalGeneration, T5Tokenizer, ConvNextConfig
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
    def __init__(self, 
                 tokenizer=False, 
                 pretrained_encoder="convnext-tiny", 
                 pretrained_decoder="t5-small", 
                 use_connector=False,
                 max_phrase_length=1024, 
                 min_phrase_length=5, 
                 num_beams=4,
                 no_repeat_ngram_size=2, 
                 do_sample=True, 
                 temperature=1,
                 top_k=50,
                 top_p=1,
                 repetition_penalty=1):

        super().__init__()

        self.tokenizer = tokenizer
        self.max_phrase_length = max_phrase_length
        self.min_phrase_length = min_phrase_length
        self.conv_only = False
        self.use_connector=use_connector

        self.num_beams=num_beams
        self.no_repeat_ngram_size=no_repeat_ngram_size
        self.do_sample=do_sample
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p
        self.repetiton_penalty=repetition_penalty



        if pretrained_decoder == "t5-small":
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-small")
        elif pretrained_decoder == "t5-base":
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-base")
        elif pretrained_decoder == "t5-large":
            self.decoder = T5ForConditionalGeneration.from_pretrained("t5-large")

        if pretrained_encoder == "convnext-tiny":
            self.encoder = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
            in_channels=768
        elif pretrained_encoder == "convnext-tiny-raw":
            configuration = ConvNextConfig(is_encoder_decoder=False)
            self.encoder = ConvNextModel(configuration)
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
        elif pretrained_encoder == "convnext-xlarge":
            self.encoder = ResNetModel.from_pretrained("facebook/convnext-xlarge-224-22k-1k")
            in_channels = 2048
        elif pretrained_encoder == "conv":

            self.conv_only = True
            self.encoder = nn.Conv2d(in_channels=3, 
                                out_channels=self.decoder.config.d_model,
                                kernel_size=135,
                                stride=10,
                                padding=0)
        
        
        if self.conv_only == False:
            self.connect_enc_dec = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=self.decoder.config.d_model,
                                            kernel_size=1,
                                            stride=1,
                                            padding=1)
        
    def n_params(self):
        encoder_params = get_n_params(self.encoder)
        decoder_params = get_n_params(self.decoder)
        if self.conv_only:
            connector_params = 0
        else:
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
        if self.conv_only:
            out = self.encoder(out)
        else:
            ...
            out = self.encoder.forward(out, return_dict = False)[0]
            if self.use_connector:
                out = self.connect_enc_dec(out)
        if self.use_connector:
            out = out.permute(0, 2, 3, 1).reshape(-1, 81, self.decoder.config.d_model)
        else:
            out = out.permute(0, 2, 3, 1).reshape(out.shape[0], -1, self.decoder.config.d_model)
      

        if len(args)>1:
            labels = args[1]
            out = self.decoder(inputs_embeds=out, labels=labels, return_dict=True)

        else:

            out = self.decoder.generate(inputs_embeds=out,
                                    num_beams=self.num_beams,
                                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                                    do_sample=self.do_sample,
                                    min_length=self.min_phrase_length,
                                    max_length=self.max_phrase_length,
                                    early_stopping=True,
                                    temperature=self.temperature,
                                    top_k=self.top_k,
                                    top_p=self.top_p,
                                    repetition_penalty=self.repetiton_penalty)
        return out



if __name__ == "__main__":

    from torchsummary import summary

    def translate_encoded_ids(encoded_ids_list, tokenizer):
        phrases = []
        for encoded_ids in encoded_ids_list:
            decoded_ids = tokenizer.decode(encoded_ids, skip_special_tokens=True)
            phrases.append(decoded_ids)
        return phrases

    class ConvNextDebugger(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model 
        
        def forward(self, input):
            out = self.model.forward(input, return_dict=False)
            out = out[0]
            return out 
        
        def __call__(self, input):
            out = self.forward(input)
            return out 
    

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = ConvNext2T5Model(tokenizer=tokenizer, pretrained_encoder="convnext-tiny", pretrained_decoder="t5-small", use_connector=True)
    model = model.to(device)
    model.n_params()
    input = torch.rand(size=(1,3,224,224))

    print(input.shape)
    #out = model(input)
    #print(out)
    #translated_out = translate_encoded_ids(out, tokenizer)
    #print(translated_out)

    import os
    print(os.getcwd())
    print(os.listdir())
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images")
    from data_folder.medical_datasets import RocoDataset
    from torch.utils.data import Dataset, DataLoader

    roco_path = "/Users/pdcos/Documents/Mestrado/IA025/Projeto_Final/Code/caption_medical_images/dataset_structure/roco-dataset"
    valid_dataset = RocoDataset(roco_path=roco_path, mode="validation", caption_max_length=64, tokenizer=tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)
    img, cap_in, cap_target, keywords, img_name = next(iter(valid_loader))
    out = model(img)
    print(out)
    #model_debug = ConvNextDebugger(model)
    #summary(model, (3,224,224))

