SW4CK(SW4 Curvilinear Kernels) consists of 5 stencil evaluation kernels that account for
~50% of the solution time in [**SW4**](https://github.com/geodynamics/sw4).

In branch(low_register) , kernels 1 & 5 are split into 7 parts to control register usage. Registers used drops to 170 each
compared to 256+spills for the original version
Documentation
----------------

Build and run instuctions for Nvidia and AMD GPUs
are available in INSTALL.txt. Sample outputs from several
machines at Livermore Computing, LLNL are available in the sample_outputs directory

Community
------------------------

SW4CK is an open source project.  Questions, discussion, and
contributions are welcome. Contributions can be anything from ports to new 
architectures to bugfixes or documentation.

Contributing
------------------------

Releases
--------

Code of Conduct
------------------------
Please note that SW4CK has a
[**Code of Conduct**](.github/CODE_OF_CONDUCT.md). By participating in
the SW4CK community, you agree to abide by its rules.


Authors
----------------
Many thanks go to SW4's [contributors](https://github.com/geodynamics/sw4/graphs/contributors).

SW4CK was created by Ramesh Pankajakshan, pankajakshan1@llnl.gov.

License
----------------

SW4CK is distributed under the terms of GPL, v2.0 license.

All new contributions must be made under the terms of the GPL, v2.0 license.

SPDX-License-Identifier: GPL-2.0-only

LLNL-CODE-821241
