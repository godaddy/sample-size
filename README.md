# sample-size

This python project is a helper package that uses power analysis to calculate required sample size for any experiment.

## Script Usage Guide
```
pip install sample-size
run-sample-size
```

## Script Constraints
* This package supports 
  * Single metric per calculator for now
  * Metric types: Boolean, Numeric, and Ratio
* Default statistical power (80%) is used in `run-sample-size` all the time
* Input constraints
  * alpha: (0, 0.3]
  * probability (Boolean Metric): (0, 1)
  * variance (Numeric and Ratio Metrics): [0, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;&plus;\infty" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\small&space;&plus;\infty" title="\small +\infty" /></a>)


## Contributing

All contributors and contributions are welcome! Please see the [contributing docs](CONTRIBUTING.md) for more information.