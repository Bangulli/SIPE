# Stain Invariant Patch Encoder (SIPE)

Embeddings of histopathology are strongly affected by the staining compound used
SIPE aimes to reduce the staining bias by focusing on morphology instead.
This is achieved by a contrastive loss that rewards attraction according to staining compound in a subsection of the feature vector and punishes it in the rest, this loss is combined with two reconstruction losses: image rec, which uses the full vector to reconstruct the full image, and morph rec which uses the stain-less subsection of the vector to reconstruct a canny mask (not yet decided) of the image.

The idea is that this way color invariant mophological features can be extracted from any image, invariant to staining.