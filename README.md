# RECONSTRUCTION VNET FOR DATA QUALITY - README

A reconstruction VNet to reconstruct original MRI volumes from a bad quality version of themselves. This is achieved by applying transformations to the original image that worsen its quality. These transformations try to simulate real artifacts or inhomogenities that might appear in MRI volumes. 

An example of an original MRI frame and its bad quality transformation:

<!---
![Example Preview](https://github.com//fabibombo/rec_vnet_quality/blob/main/pictures/bad_quality_example.png?raw=true)
-->

<img src="https://github.com/fabibombo/rec_vnet_quality/blob/main/pictures/bad_quality_example.png?raw=true" alt="Example Preview" width="200">

Currently under development:
- Add training script.
- *transforms_bq.py* under revision.
- Containerization.

By David Vallmanya Poch

Contact: davidvp12@gmail.com
