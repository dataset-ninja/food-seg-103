Dataset **FoodSeg103** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzM2MjZfRm9vZFNlZzEwMy9mb29kc2VnMTAzLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogIjFMZk1xREpOd25kSHJvRGJSV2FmNURlN3RJV3JzTE5EcWVRK3JDa0d0S1E9In0=)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='FoodSeg103', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip).