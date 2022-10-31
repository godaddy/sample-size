v3.0.0 - 2022-10-31
----------------------

Feature:
* [#61](https://github.com/godaddy/sample-size/pull/61) - support sample size calculation for one-sided test and update documentation


v2.0.3 - 2022-09-27
----------------------

Change:
* [#52](https://github.com/godaddy/sample-size/pull/52) - optimize the use of random state parameter in order to resolve the long response time issue

v2.0.2 - 2022-09-21
----------------------

Fix:

* [#48](https://github.com/godaddy/sample-size/pull/48) - Add random state parameters to have all simulation functions generating fixed output given a fixed input


v2.0.1 - 2022-08-15
----------------------

Other changes:

* [#42](https://github.com/godaddy/sample-size/pull/42) - Performance improvements
* [#44](https://github.com/godaddy/sample-size/pull/44) - Update internal parameters for binary search

v2.0.0 - 2022-06-08
----------------------

Feature:

* [#35](https://github.com/godaddy/sample-size/pull/35) - Add implementation for sample size calculator which supports multiple metrics and/or variants
  * [#30](https://github.com/godaddy/sample-size/pull/30) User experience change: prompt a question on number of variants and enable multiple metrics registration


v1.0.0 - 2021-12-14
----------------------

Features:

* [#4](https://github.com/godaddy/sample-size/pull/4) - Add implementation for sample size calculator which supports a single metric calculation.
* [#6](https://github.com/godaddy/sample-size/pull/6) - Add script to run sample size calculation locally.

Other changes:

* [#1](https://github.com/godaddy/sample-size/pull/1) - Setup the repo with CI using github actions
* [#3](https://github.com/godaddy/sample-size/pull/3) - Setup release workflow and instructions
