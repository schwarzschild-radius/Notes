# Creating Effective Container Images
## Layers
* Layers are building blocks of docker images. The number of size is directly proportional to the size of the image.
* The layers of an image can be found by using `docker history [image name \ ID]`.
* There are two types of layers
    1. Base Image layer - immutable layers
    2. Thin RW layers

## Some Best Practises to avoid image size explosion
* Use shared base images - This is possibly where the most size of the container resides
* Limit data written to the container layer while docker build. the build context is important.
* chain `RUN` statements. Combine `RUN` statments into one so that it doesn't create additonal layers
* Prevent cache misses at build
* Do Multi-stage builds
* Use scratch images???
* Remove unnecessary cache like `pip cache`.
* Use `docker image prune -a` and `docker system prune -a` to clean up after building an image