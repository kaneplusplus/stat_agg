stat\_agg
===

Statistical aggregates with Redis.

Description
---

The goal of the stat\_agg package is to:

1. Provide a complete suite of statistical aggregators that maximize ensemble
prediction accuracy.
2. Manage computational complexity by statistical and machine learning 
algorithms by allowing users to specify subsets of training data during 
the learning step.
3. Manage ensemble in a way that is distributed and elastic. New learners
can be added to an ensemble at any time.
4. Provide fault-tolerance when one or more of the learners suddenly becomes
unavailable.

Requirements
---

The stat\_agg package requires Python (tested on version 2.7), the 
laputa Python package, the py-sdm Python package, and the scikit-learn Python 
package (version 0.13 or above).

Installing stat\_agg
---

The easiest way to install stat\_agg is to use pip from within a shell:

```bash
> pip install -e git+https://github.com/kaneplusplus/stat_agg.git#egg=stat_agg
```

This package can also be installed with pip and the following shell commands:

```bash
> python setup.py sdist
> pip install dist/stat_agg-0.1.tar.gz
```

Using stat\_agg
---

The easiest way to use stat\_agg is from a shell. Assume that feats_train.h5, 
feats_validate.h5, and feats_test.h5 are the training, validation, and 
testing features that have been generated with the py-sdm package. Then
the accuracy of an ensemble of 4 sdm's, where each learner uses 25% of
the training data, and uses 4 processor threads can be found with the following 
command:

```bash
> stat_agg_bench -t feats_train.h5 -l feats_validate.h5 -r feats_test.h5 -s 0.25 -w 4 -p 4 
```

Support
---

stat_agg is supported on Python version 2.7

The development home of this project can be found at: [https://github.com/kaneplusplus/stat\_agg](https://github.com/kaneplusplus/stat\_agg)

The package currently supports support distribution machines, implemented in
the py-sdm package. However, new learners are easy to add and more will
be supported in the near future.

The package currently supports the following statistical aggregators:
- Max vote
- Min vote
- Average vote
- Minimum variance
with the following aggregators planned for implementation:
- Minimum L2
- Least squares

Contributions are welcome.

For more information contact Michael Kane at [kaneplusplus@gmail.com](kaneplusplus@gmail.com).

