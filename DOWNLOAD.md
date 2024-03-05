Dataset **FoodSeg103** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/M/9/hT/o2J96XRb4w17q841xlrC2Ugk5DLBWnnhtTgarxwITNBgWy7MClzUeilcS8yDRlSpa8D1BuzTgLj9PFfGfB3gE6y2YExJh9JLQTFyJdpMjl4t5GlXbrWyx6pX64gY.tar)

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