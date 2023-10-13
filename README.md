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

```bibtex
@inproceedings{ijcai2023p0448,
  title     = {Recognizable Information Bottleneck},
  author    = {Lyu, Yilin and Liu, Xin and Song, Mingyang and Wang, Xinyue and Peng, Yaxin and Zeng, Tieyong and Jing, Liping},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {4028--4036},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/448},
  url       = {https://doi.org/10.24963/ijcai.2023/448},
}
```

## License
MIT
