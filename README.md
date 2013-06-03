elr
===

Ensemble Learners with Redis.

Description
---

The goal of elr is to:
1. Provide a framework for building ensembles of learners for classification
and regression challenges.
2. Manage computational complexity by statistical and machine learning 
algorithms by allowing users to specify subsets of training data during 
the learning step.
3. Manage ensemble in a way that is distributed and elastic. New learners
can be added to an ensemble at any time.
4. Provide fault-tolerance when one or more of the learners suddenly becomes
unavailable.
5. Provide a complete suite of statistical aggregators to maximize ensemble
prediction accuracy.

Requirements
---

The elr package requires Python (tested on version 2.7), the cnidaria Python
package, the py-sdm Python package, and the scikit-learn Python package 
(version 0.13 or above).

Installing elr
---

The easiest way to install elr is to use pip from within a shell:

```bash
> pip install -e git+https://github.com/kaneplusplus/elr.git#egg=elr
```

This package can also be installed with pip and the following shell commands:

```bash
> python setup.py sdist
> pip install dist/elr-0.1.tar.gz
```

Using elr
---

The easiest way to use elr is from a shell. Assume that feats_train.h5, 
feats_validate.h5, and feats_test.h5 are the training, validation, and 
testing features that have been generated with the py-sdm package. Then
the accuracy of an ensemble of 4 sdm's, where each learner uses 25% of
the training data, and uses 4 processor threads can be found with the following 
command:

```bash
> elr_bench -t feats_train.h5 -l feats_validate.h5 -r feats_test.h5 -s 0.25 -w 4 -p 4 
```

Support
---

elr is supported on Python version 2.7

The development home of this project can be found at: [https://github.com/kaneplusplus/elr](https://github.com/kaneplusplus/elr)

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

