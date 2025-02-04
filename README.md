# TunedSAMModel

SAM segmentation model but I tuned/improved it:

-resizing, GPU, etc,
changed the device assignment to use GPU if available.
Ensured the image resizing function is efficient and does not introduce unnecessary overhead
Optimized the mask writing process to reduce I/O overhead.
Reran on Our curly hair kaggle datasets

