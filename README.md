SW4CK(SW4 Curvilinear Kernels) consists of 5 stencil evaluation kernels that account for
~50% of the solution time in [**SW4**](https://github.com/geodynamics/sw4).

[**MERGE**] branch.

Kernels 2,3 & 4 merged into 1 kernel. Runtime ~3X slower:

/sw4ck  sw4ck.in
HIP(5.4.22804)
AMD unroll fix enabled
Magic sync enabled

Reading from file sw4ck.in
Launching sw4 kernels

Kernel 1 time 6.03074
Kernel 2 time 19.6401
Kernel 5 time 6.04994

Total kernel runtime = 31 milliseconds (31809 us ) 

MIN = -1.5735458151501329865e-05
MAX = 0.010639705916308216105

Norm of output 0x1.941a40aec142ep+7
Norm of output 202.0512747393526638
Error = 0 %


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
