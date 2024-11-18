from modelling import ClipTextEmbedder, T5Embedder, load_autoencoder, load_flux


class FluxGenerator:
    def __init__(self):
        self.ae = load_autoencoder(
            "black-forest-labs/FLUX.1-schnell",
            "ae.safetensors",
            scale_factor=0.3611,
            shift_factor=0.1159,
        )
        self.flux = load_flux()
        self.t5 = T5Embedder()
        self.clip = ClipTextEmbedder()
