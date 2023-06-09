# Recognizable Information Bottleneck

This is the official repo of **Recognizable Information Bottleneck** (RIB).

## Requirement

The code is implemented using PyTorch framework. Use `pip` or `conda` to install all dependencies:
```shell
pip install -r requirements.txt
```
or
```shell
conda install --file requirements.txt
```
## Usage

Run the following command to get all the experimental results in the paper:
```shell
./run.sh `which python` <your log dir> <your data dir>
```
It will automatically invoke all the GPUs on the machine to run the experiments. 
It took about 16 hours to run through on our machine (8 × NVIDIA RTX A4000).

## Citation

```plain
@article{lyu2023recognizable,
  title={Recognizable Information Bottleneck},
  author={Lyu, Yilin and Liu, Xin and Song, Mingyang and Wang, Xinyue and Peng, Yaxin and Zeng, Tieyong and Jing, Liping},
  journal={arXiv preprint arXiv:2304.14618},
  year={2023}
}
```

## License
MIT