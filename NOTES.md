# Te edge error note

The extremely small SOL uncertainties may give the outermost Te points disproportionate weight, forcing your mtanh tail downward. Adding a fixed or relative error component could reduce that weight and allow the fitted tail to rise toward Davide’s curve.

Approach: error perturbation of the edge Te data, implemented in `augment_measurement_errors` ([preprocess.py](src/ts_profile_fitter/preprocess.py)).

Formulation:
$$
\sigma_{\mathrm{new}} = \sqrt{\sigma_{\mathrm{original}}^2 + \sigma_{\mathrm{fixed}}^2 + \left(f_{\mathrm{relative}} T_e\right)^2}
$$
