# V7 Gale 

This framework is an evolved fork of [DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments](https://diva-dia.github.io/DeepDIVAweb/index.html).
The major differences are the full adoption of an object oriented programming design, the polishing of the workflow, the introduction of an optimized inference-use case and a better isolation between the tasks.

This work has been conducted during an internship at V7, London, UK.

## Additional resources

- [DeepDIVA Homepage](https://diva-dia.github.io/DeepDIVAweb/index.html)
- [Tutorials](https://diva-dia.github.io/DeepDIVAweb/articles.html)
- [Paper on arXiv](https://arxiv.org/abs/1805.00329) 

## Citing us

If you use our software, please cite our paper as:

``` latex
@inproceedings{albertipondenkandath2018deepdiva,
  title={{DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments}},
  author={Alberti, Michele and Pondenkandath, Vinaychandran and W{\"u}rsch, Marcel and Ingold, Rolf and Liwicki, Marcus},
  booktitle={2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR)},
  pages={423--428},
  year={2018},
  organization={IEEE}
}
```

## License

Our work is on GNU Lesser General Public License v3.0

## Getting started

In order to get the framework up and running it is only necessary to clone the latest version of the repository:

``` shell
git clone https://github.com/v7labs/Gale.git
```

Run the script:

``` shell
bash setup_environment.sh
```

Reload your environment variables from `.bashrc` with: `source ~/.bashrc`

Some runners require additional packages. To install them, simply run the `extend_environment.sh` script in the folder
of the respective runner.
