VQGAN-LEDGM-SegAD
Anomaly Detection using a new 3-step reconstruction process and architecture. Using a VQGAN-LEDGM hybrid with autoregressive transformers and SegAD on the CableInspect-AD and MVTec-3D-AD datasets.

Anomaly maps of both CableInspectAD & MVTec3D-AD are located in the Anomaly Maps folder.

Here is all the code I developed for my project. pro_curve_util.py, generic_util.py were taken from the MVTec_3d_evaluation_code folder with minimal changes.
The get_border function in CreateCableDataset.py was developed from the folks who create the CableInspectAD Dataset. I altered the function slightly to acquire the images I used in my setup.   
