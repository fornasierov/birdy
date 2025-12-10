# birdy
Tiny audio processor and classifier focused on bird species identification.

# Install

## On Fedora

1. Install the Intel compute runtime to enable XPU usage on PyTorch:
```
sudo dnf install intel-compute-runtime oneapi-level-zero intel-media-driver
``` 
2. Add permissions to let our user utilize the XPU:
```
sudo usermod -aG render $USER
```
3. Restart computer
4. Verify with `clinfo | grep "Device Name"`. Result should be something like:
```
    Device Name                                   Intel(R) Arc(TM) Graphics
    Device Name                                   Intel(R) Arc(TM) Graphics
    Device Name                                   Intel(R) Arc(TM) Graphics
    Device Name                                   Intel(R) Arc(TM) Graphics
```
