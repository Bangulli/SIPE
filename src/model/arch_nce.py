
#########################################################################################################################################
# The problem is that the specified part of the vector collapses to 0. No solution yet, maybe using BYOL
class H0_mini_for_InfoNCE(nn.Module):
    def __init__(self, base_model="hf-hub:bioptimus/H0-mini", emb_stain_size=64, device='cuda:0', clr_mode='project'):
        super().__init__()
        self.device = device
        self.to_pil = ToPILImage()
        self.emb_stain_size = emb_stain_size ## how many elements of the feature vector should contain staining information
        num_features = 768 ## embedding size, depends on base_model
        self.backbone, self.transform = get_encoder_and_transforms(base_model) ## backbone model
        print('Backbone built!')
        self.image_decoder = get_decoder(base_model, 3, num_features) ## decoder for image recon
        print('Image Decoder built!')
        self.morph_decoder = get_decoder(base_model, 1, num_features-emb_stain_size) ## decoder for morpholgy recon -> canny mask
        print('Morph Decoder built!')
        self.projector = FullProjector(self.emb_stain_size, num_features-self.emb_stain_size) if clr_mode=='project' else FullNormalizer(self.emb_stain_size, num_features-self.emb_stain_size)
        print('Projector built!')
        self.to(self.device)
        
    def forward(self, batch):
        return self.backbone(batch['image'].to(self.device))[:, 0, :]
    
    def loss(self, batch, loss, logger=None):  
        emb = self.backbone(batch['image'].to(self.device))[:, 0, :]
        
        ## project and split
        subsec_stain, subsec_morph = self.projector(emb)
        ## recon
        rec_img = self.image_decoder(emb.unsqueeze(-1).unsqueeze(-1))
        rec_morph = self.morph_decoder(emb[:, self.emb_stain_size:].unsqueeze(-1).unsqueeze(-1))
        
        return loss(batch, subsec_stain, subsec_morph, rec_img, rec_morph, self.device, logger)

    def recon_image(self, emb, transform=None):
        rec = self.image_decoder(emb)
        if transform is not None: rec = transform(rec)
        return rec
    
    def recon_image_PIL(self, emb, transform=None):
        return self.to_pil(self.recon_image(emb, transform).squeeze(0))
    
    def recon_morph(self, emb, transform=None):
        rec = self.morph_decoder(emb[:, self.emb_stain_size:])
        if transform is not None: rec = transform(rec)
        return rec
        
    def recon_morph_PIL(self, emb, transform=None):
        return self.to_pil(self.recon_morph(emb, transform).squeeze(0))
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'H0-mini_from_%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.mkdir(pth)
        torch.save(self.backbone.state_dict(), pth/'backbone.pth')
        torch.save(self.image_decoder.state_dict(), pth/'image_decoder.pth')
        torch.save(self.morph_decoder.state_dict(), pth/'morph_decoder.pth')
        torch.save(self.projector.state_dict(), pth/'projector.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.backbone.load_state_dict(torch.load(pth/'backbone.pth'))
        self.image_decoder.load_state_dict(torch.load(pth/'image_decoder.pth'))
        self.morph_decoder.load_state_dict(torch.load(pth/'morph_decoder.pth'))
        self.projector.load_state_dict(torch.load(pth/'projector.pth'))
