from stage3.classify_helper_functions import * 


class ClipModel:
    ''' ClipModel class to get all clip model , preprocess and device '''
    def __init__(self, clip_model: str = 'ViT-B-32', pretrained:str = 'openai'):
        
        self.clip_model = clip_model
        self.pretrained = pretrained
        
        self.clip , self.preprocess , self.device = get_clip(self.clip_model, self.pretrained)

