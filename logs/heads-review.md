# Independent detector-head review before training/scoring

One fresh-context read-only reviewer checked the plan, all heads code and key
dependencies. It found no important issue blocking training. Three CPU tests
passed during review; the suite was then expanded to five, all green.

Confirmed: complete training-span coordinates preserve logistic L2 regularization,
fit-only centering/cross-kernel transformation, standard hinge SVC with free
intercept, fit-only RBF median,36 settings×10 folds, train-only selection and five
final heads. Unique head_analysis name avoids the previous import collision.

Additional RED→GREEN negative tests cover incomplete cache, wrong source/lock,
and deterministic oneSE selection. Actual cache shape/dtype and extraction Git
source map are also checked. The original extraction source is retained in its
historical commit; later validation additions do not change cached vectors.

The review did not execute GPU tasks or inspect heldout metrics. Real full-vector
pilot equivalence, convergence,360 folds and evaluation remain runtime gates.
