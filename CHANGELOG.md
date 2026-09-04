# Changelog

## [0.12.0](https://github.com/cubiepy/cubie/compare/v0.11.1...v0.12.0) (2026-09-04)


### Features

* **backend:** Newton and Krylov iteration loops take separate unroll flags ([#910](https://github.com/cubiepy/cubie/issues/910)) ([1391bf3](https://github.com/cubiepy/cubie/commit/1391bf35b9ec7bef9e957ebe7874cfe17914a142))


### Bug Fixes

* **cache:** radau gamma host-invariant, index writes retried, mutable fixtures restore system flags ([#907](https://github.com/cubiepy/cubie/issues/907)) ([e801689](https://github.com/cubiepy/cubie/commit/e8016899b9a4bf11ec5df9fcf5c57b989d70d4e9))

## [0.11.1](https://github.com/cubiepy/cubie/compare/v0.11.0...v0.11.1) (2026-09-03)


### Bug Fixes

* **backend:** numba-cuda-mlir shims mirror the open fork PRs; mlir wheel floor moved to 0.5.1.1 ([#902](https://github.com/cubiepy/cubie/issues/902)) ([5bea622](https://github.com/cubiepy/cubie/commit/5bea62258ad54179db15402c346c9fcfc4ff15b9))
* **cuda_simsafe:** devfunc_returns_nonfloat reads compiled overload return types ([#895](https://github.com/cubiepy/cubie/issues/895)) ([2354eeb](https://github.com/cubiepy/cubie/commit/2354eeb2176e635c0998114750e5db6b170d6113))

## [0.11.0](https://github.com/cubiepy/cubie/compare/v0.10.1...v0.11.0) (2026-09-02)


### Features

* **backend:** per-group selective unrolling through llvm hints ([#881](https://github.com/cubiepy/cubie/issues/881)) ([750bc42](https://github.com/cubiepy/cubie/commit/750bc4260e8f81a76f9ee6f5dafee3bd9b517d1a))
* **backend:** unroll_converged_exits group hints the solver iteration loops; every unroll group defaults to full ([#882](https://github.com/cubiepy/cubie/issues/882)) ([df0193b](https://github.com/cubiepy/cubie/commit/df0193b16ceedb535d85c4f47f3ad743a9611b01))
* **integrators:** ELDIRK tableaus with trailing explicit stages after the DIRK Newton loop ([#894](https://github.com/cubiepy/cubie/issues/894)) ([b3f92f6](https://github.com/cubiepy/cubie/commit/b3f92f6916047bab292ff2de668b9165d8cc73ce))


### Bug Fixes

* **algorithms:** dense-predictor buffer locations passed at construction reach the predictor ([#857](https://github.com/cubiepy/cubie/issues/857)) ([ce1ce83](https://github.com/cubiepy/cubie/commit/ce1ce83122197ee74d5e6904cee15c237ed96dd1))
* **backend:** numba-cuda-mlir patches for dominator search and topographical order ([#870](https://github.com/cubiepy/cubie/issues/870)) ([b642a29](https://github.com/cubiepy/cubie/commit/b642a291e491933812c466a2dc8dc5844aae81fc))
* **buffer_registry:** aliases never overlap persistent storage at any nesting depth ([#861](https://github.com/cubiepy/cubie/issues/861)) ([6eb25ac](https://github.com/cubiepy/cubie/commit/6eb25acf571c9efb5e8e8521881ae7bbbbe211ed))
* **integrators:** DIRK explicit stage information centralised to the tableau ([#889](https://github.com/cubiepy/cubie/issues/889)) ([663ac19](https://github.com/cubiepy/cubie/commit/663ac19f6ca2fe241ff81a67a400aa71eb2c1f93))
* **integrators:** error buffers compile out under fixed step control ([#887](https://github.com/cubiepy/cubie/issues/887)) ([df84d27](https://github.com/cubiepy/cubie/commit/df84d27a5769cc65507324785d4a94f4b1ab401c))
* **integrators:** Tableau coefficients return as arrays for better MLIR lowering ([#888](https://github.com/cubiepy/cubie/issues/888)) ([80f5f06](https://github.com/cubiepy/cubie/commit/80f5f0643f06fa1f99e2c6fd0319b49bcf32cf13))
* **loops:** Save and end time clamping can no longer drift from schedule ([#885](https://github.com/cubiepy/cubie/issues/885)) ([0abd859](https://github.com/cubiepy/cubie/commit/0abd859f72cb20fda603e584d1226d5a31da5311))
* **memory:** cross-solver allocation queue contamination prevented, dead requests deleted ([#860](https://github.com/cubiepy/cubie/issues/860)) ([6c5f2e2](https://github.com/cubiepy/cubie/commit/6c5f2e2137b34f413379cdf8b1bc060901a65006))
* **solvers:** BiCGSTAB witness vector defaults to local like every other buffer ([#873](https://github.com/cubiepy/cubie/issues/873)) ([23f2a37](https://github.com/cubiepy/cubie/commit/23f2a372d432ab8a4d611836cc071b63871ee2e8))


### Performance Improvements

* **backend:** block scheduler barriers order through the previous barrier only ([#871](https://github.com/cubiepy/cubie/issues/871)) ([e1cc2f5](https://github.com/cubiepy/cubie/commit/e1cc2f54187efe19ff9aeb7c54fbc17c73f8a593))
* **loops:** loop entry relies on the allocator zero fills ([#893](https://github.com/cubiepy/cubie/issues/893)) ([5486569](https://github.com/cubiepy/cubie/commit/5486569589373faf5fdbbbfd3f3fb2d04ec4b1f1))


### Documentation

* **tests:** ring modulator fixture docstrings state the system only ([#856](https://github.com/cubiepy/cubie/issues/856)) ([4e36534](https://github.com/cubiepy/cubie/commit/4e365348b6c73173e7142ed1678392580735d4ec))
* third-party licence notices shipped with the package ([#862](https://github.com/cubiepy/cubie/issues/862)) ([5427332](https://github.com/cubiepy/cubie/commit/54273322c59324234edc85a47b596a2784175df6))
* timing and performance measurements sized to at least two occupancy waves ([#878](https://github.com/cubiepy/cubie/issues/878)) ([63c7800](https://github.com/cubiepy/cubie/commit/63c78001187b51801f3ab13e13705c2a711b6e52))

## [0.10.1](https://github.com/cubiepy/cubie/compare/v0.10.0...v0.10.1) (2026-08-26)


### Bug Fixes

* **solvers:** iterative linear solves keep a finite stopping target when the entry norm overflows ([#847](https://github.com/cubiepy/cubie/issues/847)) ([9d8c4f1](https://github.com/cubiepy/cubie/commit/9d8c4f1319f5c7e2e0247ef959f36747291f60df))

## [0.10.0](https://github.com/cubiepy/cubie/compare/v0.9.0...v0.10.0) (2026-08-25)


### Features

* **integrators:** Buffer location heuristics updated for 0.9.0 ([#828](https://github.com/cubiepy/cubie/issues/828)) ([acaf5dd](https://github.com/cubiepy/cubie/commit/acaf5ddd969eed406d4f60372839490e678008d7))
* **integrators:** Shampine DAE initialisation solves using damped Newton ([#835](https://github.com/cubiepy/cubie/issues/835)) ([c53db37](https://github.com/cubiepy/cubie/commit/c53db37b6db93444c410333709296e8860a638cf))
* **step_control:** controller gains match PID gain concepts, raw coefficients optional ([#840](https://github.com/cubiepy/cubie/issues/840)) ([64055b4](https://github.com/cubiepy/cubie/commit/64055b4e455341fdfe8b00962f9a6531241783c9))


### Bug Fixes

* **algorithms:** rosenbrock stage_increment no longer aliases stage_store ([#837](https://github.com/cubiepy/cubie/issues/837)) ([2198625](https://github.com/cubiepy/cubie/commit/2198625e2366f470dbf9d4483de0491a05c4b1e0))
* **ci:** user-provided functions that return floats now precision-wrapped ([#830](https://github.com/cubiepy/cubie/issues/830)) ([b74a566](https://github.com/cubiepy/cubie/commit/b74a5668f43b5f616ce1a1a1f3090320be15c714))
* **codegen:** direct LU pivots are static row/column choices over structural nonzeros ([#827](https://github.com/cubiepy/cubie/issues/827)) ([cb8a02b](https://github.com/cubiepy/cubie/commit/cb8a02b45338099f3ce16d43fc380f7169a63e71))
* **codegen:** LU factorization picks same-stage pivots first for FIRK solves ([#838](https://github.com/cubiepy/cubie/issues/838)) ([27b28ac](https://github.com/cubiepy/cubie/commit/27b28ac46de8c2c5a123764002da1bfe99488c5d))
* **integrators:** DAEs default to the LU linear solver ([#825](https://github.com/cubiepy/cubie/issues/825)) ([58156ff](https://github.com/cubiepy/cubie/commit/58156ff0160473f85b9e80d70a8b8eca5cce4c9c))
* **memory:** chunk sizing keeps some headroom, staging blocks stay bounded, free memory includes in-pool free bytes ([#845](https://github.com/cubiepy/cubie/issues/845)) ([79b8ee3](https://github.com/cubiepy/cubie/commit/79b8ee30dafc6bfd4967139621d4260e62829ad9))
* **memory:** pinned-array finalizers queued without lock (stops race) ([#841](https://github.com/cubiepy/cubie/issues/841)) ([44e7565](https://github.com/cubiepy/cubie/commit/44e7565d0a15125003658169ed58eb2ea77cca59))
* **solvers:** newton no longer exits on theta-based divergence ([#839](https://github.com/cubiepy/cubie/issues/839)) ([5f8a802](https://github.com/cubiepy/cubie/commit/5f8a802c1de9a112fa8936323af8ff8437b97bb8))

## [0.9.0](https://github.com/cubiepy/cubie/compare/v0.8.1...v0.9.0) (2026-08-22)


### Features

* **batchsolving:** runs with errors now warn and offer remedies ([#820](https://github.com/cubiepy/cubie/issues/820)) ([1eadcbc](https://github.com/cubiepy/cubie/commit/1eadcbcd04298d2b44038b2384a9be19cb9d978c))
* **batchsolving:** Solver.calibrate finds the fastest solver for your system  ([#812](https://github.com/cubiepy/cubie/issues/812)) ([9dcc859](https://github.com/cubiepy/cubie/commit/9dcc859b67c19ab10bf871ec75d761cdead025f6))
* **batchsolving:** Solver.compile() compiles the batch kernel without launching ([#816](https://github.com/cubiepy/cubie/issues/816)) ([38f1edd](https://github.com/cubiepy/cubie/commit/38f1edd131cabff4b24640dc19fbdf077fe0518c))
* **integrators:** adaptive DIRK defaults to dense prediction off, and kvaerno5 and l_stable_sdirk_4 to exact Newton ([#818](https://github.com/cubiepy/cubie/issues/818)) ([9acb4e1](https://github.com/cubiepy/cubie/commit/9acb4e134f0ff270860b5ce752294fd4cf5baffa))
* **integrators:** cash-karp, dop853, and vern7 default to the Gustafsson step controller ([#819](https://github.com/cubiepy/cubie/issues/819)) ([f22794e](https://github.com/cubiepy/cubie/commit/f22794e2f1e1bcf0eb113cec07aa7029f71e3f86))
* **integrators:** consistent DAE initialisation at loop entry ([#811](https://github.com/cubiepy/cubie/issues/811)) ([4b0ac2f](https://github.com/cubiepy/cubie/commit/4b0ac2f587a0567a0b27f436bdc230cfd46e5ffc))
* **integrators:** per-tableau and per-family AlgorithmDefaults now set the best solver per algo  ([#813](https://github.com/cubiepy/cubie/issues/813)) ([17e7558](https://github.com/cubiepy/cubie/commit/17e7558786a82d3cb7a99e35e8cec625f61124d4))
* **integrators:** radau_iia_9 defaults to the BiCGSTAB linear solver with exact Newton ([#817](https://github.com/cubiepy/cubie/issues/817)) ([ca718f7](https://github.com/cubiepy/cubie/commit/ca718f7e84609f17a80373461e428bea4e294b1f))

## [0.8.1](https://github.com/cubiepy/cubie/compare/v0.8.0...v0.8.1) (2026-08-20)


### Miscellaneous Chores

* **codegen:** series preconditioner loops evaluate accumulator-free JVP work once ([#803](https://github.com/cubiepy/cubie/issues/803)) ([fe75ac9](https://github.com/cubiepy/cubie/commit/fe75ac95642c2848c575a690e19638711cd3a80d))

* **preconditioners:** Preconditioner now opt-in ([#801](https://github.com/cubiepy/cubie/issues/801)) ([d8bd203](https://github.com/cubiepy/cubie/commit/d8bd20347d1b3a5b4ae26d5d63af403bc9fd91ba))

## [0.8.0](https://github.com/cubiepy/cubie/compare/v0.7.0...v0.8.0) (2026-08-20)


### Features

* **cache:** kernel-cache entry limit defaults to zero ([#796](https://github.com/cubiepy/cubie/issues/796)) ([71e3969](https://github.com/cubiepy/cubie/commit/71e39691d24f3ea9f392944583e53287e1f871c4))
* **preconditioners:** jacobi preconditioner now has order 1, 2. ([#792](https://github.com/cubiepy/cubie/issues/792)) ([ce05d26](https://github.com/cubiepy/cubie/commit/ce05d26249919e3436bb0b18eba8c0caa4e422fb))
* **solvers:** direct LU linear solvers added, exact, inexact, and pre-factored. ([#790](https://github.com/cubiepy/cubie/issues/790)) ([a63ed59](https://github.com/cubiepy/cubie/commit/a63ed591948ae846895f1ba842ca8a007128dc32))


### Bug Fixes

* **codegen:** all cached helpers are codegenned from the same rearranged equations ([#794](https://github.com/cubiepy/cubie/issues/794)) ([633b93b](https://github.com/cubiepy/cubie/commit/633b93bd2a936d619bfb6db214b059cdb03fe7cd))
* preconditioner chaining removed ([#786](https://github.com/cubiepy/cubie/issues/786)) ([5ccea79](https://github.com/cubiepy/cubie/commit/5ccea7943c181e968d1001e4a3bccf2cb10369e2))

## [0.7.0](https://github.com/cubiepy/cubie/compare/v0.6.0...v0.7.0) (2026-08-18)


### Features

* **codegen:** constant calls, aux chains, and pow families fold at codegen ([#787](https://github.com/cubiepy/cubie/issues/787)) ([9d4debb](https://github.com/cubiepy/cubie/commit/9d4debb75e2f09d61abba209d2bd21509bba2d01))

## [0.6.0](https://github.com/cubiepy/cubie/compare/v0.5.0...v0.6.0) (2026-08-17)


### Features

* **codegen:** constant values converted to literals in codegen ([#769](https://github.com/cubiepy/cubie/issues/769)) ([79e2cd3](https://github.com/cubiepy/cubie/commit/79e2cd30d6e23954762e27174055cf844321d5cc))
* **codegen:** equal-branch selects collapse to the branch value ([#774](https://github.com/cubiepy/cubie/issues/774)) ([ed457ed](https://github.com/cubiepy/cubie/commit/ed457ed0fdb7d6bab85708bec0de8c4bd5dd2ef9))
* **codegen:** Piecewise prints as branchless selp selections ([#762](https://github.com/cubiepy/cubie/issues/762)) ([56903a0](https://github.com/cubiepy/cubie/commit/56903a0d156d7d7bdd4575cc4f3ee2100a329b67))
* numba-cuda-mlir scheduler added to complement codegen order ([#768](https://github.com/cubiepy/cubie/issues/768)) ([ee5d302](https://github.com/cubiepy/cubie/commit/ee5d302a0dc916990f3adb45a98e6d980585683c))
* **step_control:** ERK adaptive defaults use the integral controller at kp=1.2 ([#763](https://github.com/cubiepy/cubie/issues/763)) ([abb6584](https://github.com/cubiepy/cubie/commit/abb658417ff421541d4287649feb38eb3ac5c0e8))


### Bug Fixes

* **algorithms:** DIRK and FIRK step closures bind every capture ([#783](https://github.com/cubiepy/cubie/issues/783)) ([8218f34](https://github.com/cubiepy/cubie/commit/8218f344735e8654c99cc0b2d038a74159cb8ded))
* **backend:** array-literal shim removed; mlir wheel floor moved to 0.4.2.3 ([#782](https://github.com/cubiepy/cubie/issues/782)) ([cdc1d77](https://github.com/cubiepy/cubie/commit/cdc1d779a77ca924a1c897c92d23eded6148d5b5))
* **codegen:** aux-cache tie-breaks key on symbol names ([#766](https://github.com/cubiepy/cubie/issues/766)) ([253dac4](https://github.com/cubiepy/cubie/commit/253dac4e4506ec1caa4fb92c055f68366155b175))
* **codegen:** mass matrix system simplified to bool diagonal flags since it's auto-generated ([#779](https://github.com/cubiepy/cubie/issues/779)) ([50120fb](https://github.com/cubiepy/cubie/commit/50120fb5d4b8c7e620ab47eeca36fa87ea17a075))
* **integrators:** singular-mass systems default to Jacobi + BiCGSTAB  ([#765](https://github.com/cubiepy/cubie/issues/765)) ([b112f0e](https://github.com/cubiepy/cubie/commit/b112f0e7303d94d8b7705c06cf3b7f55b7ca8e38))
* **odesystems:** all system arrays (states, parameters, obs, constants) alphabetically ordered ([#776](https://github.com/cubiepy/cubie/issues/776)) ([4ad2532](https://github.com/cubiepy/cubie/commit/4ad25327649f1818604bae600aa2d5a8b7821052))
* **time_logger:** duplicate time counting in time_logging removed ([#778](https://github.com/cubiepy/cubie/issues/778)) ([2e497e3](https://github.com/cubiepy/cubie/commit/2e497e35e0445155a5605c89b391c7872bcd6eae))

## [0.5.0](https://github.com/cubiepy/cubie/compare/v0.4.0...v0.5.0) (2026-08-13)


### Features

* **algorithms:** optional smoothed embedded error estimate ([#740](https://github.com/cubiepy/cubie/issues/740)) ([be339ea](https://github.com/cubiepy/cubie/commit/be339ea1b559c6e3e20297c6f1f56af24aaecc97))
* **algorithms:** radau IIA 3 and 9 tableaus, adaptive gauss-legendre 4 ([#749](https://github.com/cubiepy/cubie/issues/749)) ([822ab48](https://github.com/cubiepy/cubie/commit/822ab4862ca782e2d76229b207f07bf9da5f1daf))
* ftz compile flag, unflushed time narrowing, and upstreamed-shim removal ([#741](https://github.com/cubiepy/cubie/issues/741)) ([edbbe03](https://github.com/cubiepy/cubie/commit/edbbe031a9b8f8bc2f98ffe6dcc80d4068972170))
* linear solvers skip initial zero-guess evaluations ([#745](https://github.com/cubiepy/cubie/issues/745)) ([0d4ab8c](https://github.com/cubiepy/cubie/commit/0d4ab8ccaba8c8806f0f41217328adc0b4cac843))
* Operation ordering strategies to added to codegen module ([#744](https://github.com/cubiepy/cubie/issues/744)) ([c31c23b](https://github.com/cubiepy/cubie/commit/c31c23b70f1b3183147d0a7d71b50294dddbeea6))


### Bug Fixes

* **compat:** numba-cuda array lowerings patched for numpy 2.5 ([#760](https://github.com/cubiepy/cubie/issues/760)) ([3c6db08](https://github.com/cubiepy/cubie/commit/3c6db083ddf7b6c333ede75a320cc1b92eac9305))
* **integrators:** error smoothing linearized at the correct state ([#746](https://github.com/cubiepy/cubie/issues/746)) ([c6abea0](https://github.com/cubiepy/cubie/commit/c6abea0cadf552db0ea51c7f3ff70fa5bd3ecc41))
* **integrators:** save_last final step lands exactly on t_end ([#754](https://github.com/cubiepy/cubie/issues/754)) ([66c2815](https://github.com/cubiepy/cubie/commit/66c281594a1bc0dd55792d5e1525bfe9beb9e985))
* **integrators:** step-size control driven by the embedded-pair order ([#743](https://github.com/cubiepy/cubie/issues/743)) ([0aba17f](https://github.com/cubiepy/cubie/commit/0aba17f6a2b3fbb1d63975f39fb11c8379b52b9c))
* **step_control:** rejected steps don't respect the control deadband; default deadband off ([#759](https://github.com/cubiepy/cubie/issues/759)) ([e84786f](https://github.com/cubiepy/cubie/commit/e84786f2b719e8d39e6ed486ee34d3c84c9d89ec))

## [0.4.0](https://github.com/cubiepy/cubie/compare/v0.3.4...v0.4.0) (2026-08-06)


### Features

* Step controllers accept f(order) for control gains, Newton Solver rtol is floored at 4*eps ([#719](https://github.com/cubiepy/cubie/issues/719)) ([80c4744](https://github.com/cubiepy/cubie/commit/80c47448284a2a45e1018418ae51ee0ea1094214))


### Bug Fixes

* a missing device and a failed writeback no longer stall a solve ([#712](https://github.com/cubiepy/cubie/issues/712)) ([7bf9761](https://github.com/cubiepy/cubie/commit/7bf97618eef967b46287ac67c43da92d94877173))
* **batchsolving:** driver evaluation seeded from the owned interpolator at kernel construction ([#736](https://github.com/cubiepy/cubie/issues/736)) ([44cdf58](https://github.com/cubiepy/cubie/commit/44cdf582c267c7154904ac3cd9ed384a4144445c))
* **ci:** bake finalizes with EC2Launch reset only ([#726](https://github.com/cubiepy/cubie/issues/726)) ([04f7cb6](https://github.com/cubiepy/cubie/commit/04f7cb6b9b2f742c5f966bc3c87946523b001b46))
* **ci:** bake OIDC session lasts two hours ([#710](https://github.com/cubiepy/cubie/issues/710)) ([9129359](https://github.com/cubiepy/cubie/commit/912935987c2b2c5956929106f1d6c0dcb9220193))
* **ci:** bake sysprep generalizes without shutting down the spot builder ([#722](https://github.com/cubiepy/cubie/issues/722)) ([0bfe0e8](https://github.com/cubiepy/cubie/commit/0bfe0e8b7d8680ffa75078d4a128ac058f0b96c1))
* **ci:** dashboard prices each leg in its availability zone's region ([#725](https://github.com/cubiepy/cubie/issues/725)) ([8af4e34](https://github.com/cubiepy/cubie/commit/8af4e341dd45db0d5328ecece30890009f065723))
* **fleet:** deployer policy covers the security-group-rule ARN leg ([#723](https://github.com/cubiepy/cubie/issues/723)) ([1c3d329](https://github.com/cubiepy/cubie/commit/1c3d3298426310200b6214de53e68114483be1b0))
* **integrators:** per-state tolerances size by n across tiled FIRK norms ([#728](https://github.com/cubiepy/cubie/issues/728)) ([415fa89](https://github.com/cubiepy/cubie/commit/415fa89368c7ac8ca31ca50056e6c7ec91991705))
* **memory:** host RAM availability counts reclaimable memory on Linux ([#713](https://github.com/cubiepy/cubie/issues/713)) ([bf28dce](https://github.com/cubiepy/cubie/commit/bf28dceaffb06f71e66826644da0a0b08178ab0f))


### Documentation

* performance gate applies to PRs that touch src/ ([#730](https://github.com/cubiepy/cubie/issues/730)) ([0a2597d](https://github.com/cubiepy/cubie/commit/0a2597d5185c1ab4674d1348ab3e5093cbcbc3ca))
* README reflects current installation and capabilities ([#724](https://github.com/cubiepy/cubie/issues/724)) ([7d65bab](https://github.com/cubiepy/cubie/commit/7d65bab55723e2ac22659ee146b4e3068ccf64bd))

## [0.3.4](https://github.com/cubiepy/cubie/compare/v0.3.3...v0.3.4) (2026-08-02)


### Bug Fixes

* **batchsolving:** extend_grid_to_array tiles defaults across runs ([#696](https://github.com/cubiepy/cubie/issues/696)) ([2a56464](https://github.com/cubiepy/cubie/commit/2a56464d5fe5cc04d5f932c50561db67e01ec213))
* **ci:** bake provisioner scripts exit clean; runner version read from file metadata ([#702](https://github.com/cubiepy/cubie/issues/702)) ([4b4ba38](https://github.com/cubiepy/cubie/commit/4b4ba387687ca0992ed44ba12bcb64045a231417))
* **memory:** pinned host backing budgets against VRAM and RAM headroom ([#697](https://github.com/cubiepy/cubie/issues/697)) ([dae5ddc](https://github.com/cubiepy/cubie/commit/dae5ddc0737dcb2e64e743f4f9c052d17bf94b6c))

## [0.3.3](https://github.com/cubiepy/cubie/compare/v0.3.2...v0.3.3) (2026-07-30)


### Bug Fixes

* host-array solves transfer directly; writeback waits drain inline ([#690](https://github.com/cubiepy/cubie/issues/690)) ([c8a3fd2](https://github.com/cubiepy/cubie/commit/c8a3fd22f1109ed66431451c70d3b83646483453))

## [0.3.2](https://github.com/cubiepy/cubie/compare/v0.3.1...v0.3.2) (2026-07-29)


### Bug Fixes

* counter and accept-flag buffers are int32-typed in every placement ([#686](https://github.com/cubiepy/cubie/issues/686)) ([5569410](https://github.com/cubiepy/cubie/commit/5569410fb9ee531754b2457134567e4ce060436d))
* FIRK tableau weights are host-invariant; bicgstab breakdown test is deterministic ([#682](https://github.com/cubiepy/cubie/issues/682)) ([f4ecfdb](https://github.com/cubiepy/cubie/commit/f4ecfdb52df70053f94c87965ee0c8216d9d85b6))
* **fleet:** a leg with no archived job log renders and prices ([#679](https://github.com/cubiepy/cubie/issues/679)) ([c1ec9c1](https://github.com/cubiepy/cubie/commit/c1ec9c192c04812bfc9799b3ee1b7e1049329be1))


### Documentation

* a DISTRUST benchmark result gets one retry only ([#687](https://github.com/cubiepy/cubie/issues/687)) ([9c900a6](https://github.com/cubiepy/cubie/commit/9c900a62b917ec7ad886dbce09208ee03ed4e3d7))
* the full test suite runs before every PR ([#678](https://github.com/cubiepy/cubie/issues/678)) ([b82f0e1](https://github.com/cubiepy/cubie/commit/b82f0e190c4ae5e0e9754760a3cda08f8fc3935a))

## [0.3.1](https://github.com/cubiepy/cubie/compare/v0.3.0...v0.3.1) (2026-07-24)


### Bug Fixes

* compile settings made immutable ([#673](https://github.com/cubiepy/cubie/issues/673)) ([e3a9645](https://github.com/cubiepy/cubie/commit/e3a9645bda7e28c14b6293de9b8d1aa5089a0fe4))
* FSAL requires an explicit stage 0, DIRK tableaus validate consistency ([#667](https://github.com/cubiepy/cubie/issues/667)) ([10cc1cf](https://github.com/cubiepy/cubie/commit/10cc1cfbef6a18d98d1ef72be0e2f9c222cb238d)), closes [#663](https://github.com/cubiepy/cubie/issues/663)
* inner krylov residual-reduction stopping target dropped to rtol/100 ([#671](https://github.com/cubiepy/cubie/issues/671)) ([0b3b99f](https://github.com/cubiepy/cubie/commit/0b3b99fec31f16c33917e6bbbeb6bb2152d454a3))
* solver vector width is solver_width; n is the state count ([#676](https://github.com/cubiepy/cubie/issues/676)) ([d2bb0d3](https://github.com/cubiepy/cubie/commit/d2bb0d3c63d05e7c5ee7ccac727a0cf827055dd3))
* step controllers don't update step history after step rejection ([#669](https://github.com/cubiepy/cubie/issues/669)) ([7a5b7aa](https://github.com/cubiepy/cubie/commit/7a5b7aae04b7a5c4bf244338828b231916850c64))
* update(linear_correction_type=...) swaps the linear-solver class ([#668](https://github.com/cubiepy/cubie/issues/668)) ([67e6402](https://github.com/cubiepy/cubie/commit/67e640297ea8194b7591dad70d1a2d47a7c6195a))


### Performance Improvements

* dense FIRK stage prediction lives in a reusable predictor factory ([#656](https://github.com/cubiepy/cubie/issues/656)) ([e553e94](https://github.com/cubiepy/cubie/commit/e553e94b144c9b5cef8d37015bb1bc571aa39651))
* dense stage prediction supports repeated nodes and explicit first stages ([#659](https://github.com/cubiepy/cubie/issues/659)) ([2a97fb0](https://github.com/cubiepy/cubie/commit/2a97fb03df9e5a9aba1c01c24c4a35b63fbbb2cd))

## [0.3.0](https://github.com/cubiepy/cubie/compare/v0.2.0...v0.3.0) (2026-07-22)


### Features

* Chunking and eviction from VRAM and RAM extended to handle simultaneous and extra-large runs ([#621](https://github.com/cubiepy/cubie/issues/621)) ([3505e02](https://github.com/cubiepy/cubie/commit/3505e02e11eac9752d2760d022fa319948d3dbc7))
* **ci:** pre-compile and cache GPU tests on a CPU runner ([#638](https://github.com/cubiepy/cubie/issues/638)) ([f2b90c6](https://github.com/cubiepy/cubie/commit/f2b90c69fcb7740676feddbe1acdc72bb5303c49))
* inputs and outputs can now stay on-device ([#635](https://github.com/cubiepy/cubie/issues/635)) ([1f70c1d](https://github.com/cubiepy/cubie/commit/1f70c1dab46c2d18b70436210f98400e437745fd))
* Kvaerno3 and Kvaerno5 ESDIRK tableaus added ([#646](https://github.com/cubiepy/cubie/issues/646)) ([8f50158](https://github.com/cubiepy/cubie/commit/8f50158f88b82d39f3f2b1270b14e46b1e6ad126))
* linear solves stop on a reduction of the entry residual ([#627](https://github.com/cubiepy/cubie/issues/627)) ([6385260](https://github.com/cubiepy/cubie/commit/6385260024e4c2731bb7ef522a0f4fff0de5a02d))
* Newton convergence bounds the update error, not the residual ([#626](https://github.com/cubiepy/cubie/issues/626)) ([f5f5a83](https://github.com/cubiepy/cubie/commit/f5f5a8369fd76885c83ea13c1814d4616a607475))
* numba-cuda-mlir is the default CUDA backend; numba-cuda deprecated ([#640](https://github.com/cubiepy/cubie/issues/640)) ([bcd1e9b](https://github.com/cubiepy/cubie/commit/bcd1e9bc12a8f21b183bfa2e22192b1ae70244a0))


### Bug Fixes

* **ci:** ci precompile now covers all mlir kernels ([#653](https://github.com/cubiepy/cubie/issues/653)) ([eda2366](https://github.com/cubiepy/cubie/commit/eda2366d22ac042e9ac41f7792fb8240eacf667e))
* **ci:** coverage reporting now works correctly on new GPU runs ([#649](https://github.com/cubiepy/cubie/issues/649)) ([f695c1c](https://github.com/cubiepy/cubie/commit/f695c1cd935f1a48cf5b0c3245f337de1048b4e9))
* **ci:** precompiled kernel cache survives the CPU-to-GPU machine hop ([#645](https://github.com/cubiepy/cubie/issues/645)) ([5a5a8c3](https://github.com/cubiepy/cubie/commit/5a5a8c3bce139b18ece675a1a9a6c026755ab7ee))
* correct sdirk4 fourth-stage coefficient ([#647](https://github.com/cubiepy/cubie/issues/647)) ([727fd2a](https://github.com/cubiepy/cubie/commit/727fd2ab39e618931018d2117fcd5776f82efa00))
* driver coefficients update after interpolator settings change ([#631](https://github.com/cubiepy/cubie/issues/631)) ([72b6b06](https://github.com/cubiepy/cubie/commit/72b6b069030b535830a16631526b4961fc2e9cf8))
* ftz flag piped through mFlorid fastmath ([#643](https://github.com/cubiepy/cubie/issues/643)) ([270efb0](https://github.com/cubiepy/cubie/commit/270efb0a24a4e50315271114c6853048bf538de8))
* GPU work syncs per-process instead of globally in test suite ([#644](https://github.com/cubiepy/cubie/issues/644)) ([04c833a](https://github.com/cubiepy/cubie/commit/04c833a0f3eef8602fd2eb0f8959c191531efbc7))
* Gustafsson controller setting "gamma" disambiguated ([#658](https://github.com/cubiepy/cubie/issues/658)) ([14ad5e7](https://github.com/cubiepy/cubie/commit/14ad5e7444731f8b9e6fd32272be41c54a8fbab1))
* solver operator beta, gamma no longer collide with system parameters ([#624](https://github.com/cubiepy/cubie/issues/624)) ([f1b13cf](https://github.com/cubiepy/cubie/commit/f1b13cfc5681cd0be7c5ce0dd2a3e543c92fe254))
* solvers release GPU buffers on collection, not on the next registration ([#620](https://github.com/cubiepy/cubie/issues/620)) ([8d04c24](https://github.com/cubiepy/cubie/commit/8d04c24c60909eb4dacdfdda58ed38da3af37172))
* time-logger tests bracket durations with the measuring clock ([#637](https://github.com/cubiepy/cubie/issues/637)) ([55cdb7b](https://github.com/cubiepy/cubie/commit/55cdb7b714a518e1e6e38b852c709b26a82f39b2))
* user model symbols can no longer alias generated codegen variables ([#654](https://github.com/cubiepy/cubie/issues/654)) ([43a1c7c](https://github.com/cubiepy/cubie/commit/43a1c7cf01d7b425f8a36779c00a0e76b52b5287))


### Performance Improvements

* hash-consed IR expression engine replaces SymPy in codegen compute ([#622](https://github.com/cubiepy/cubie/issues/622)) ([bab4837](https://github.com/cubiepy/cubie/commit/bab48378a375941b11e1f522f7890c2d1c68160c))

## [0.2.0](https://github.com/ccam80/cubie/compare/v0.1.1...v0.2.0) (2026-07-15)


### Features

* measured heuristics choose buffer memory locations by system size ([#613](https://github.com/ccam80/cubie/issues/613)) ([71fbbb1](https://github.com/ccam80/cubie/commit/71fbbb1b8c270d007be722059d4909f462142975))
* MTK.jl-style structural simplification and tearing for DAE systems ([#605](https://github.com/ccam80/cubie/issues/605)) ([a0e188b](https://github.com/ccam80/cubie/commit/a0e188b9ead3bd3cdfb0532dc0a124ff12e0d114))
* Numba-cuda MLIR backend now supported (and fast/correct) ([#617](https://github.com/ccam80/cubie/issues/617)) ([7e811fd](https://github.com/ccam80/cubie/commit/7e811fd2f9046a2af1415dc234039aa603e4e393))


### Bug Fixes

* all three disk cache layers resolve one shared cache root ([#600](https://github.com/ccam80/cubie/issues/600)) ([947f745](https://github.com/ccam80/cubie/commit/947f745894ac5aa29bba8476000ca87aa0bf09b3))
* compiled-kernel cache stamps carry an environment hash ([#615](https://github.com/ccam80/cubie/issues/615)) ([ec51161](https://github.com/ccam80/cubie/commit/ec511610d4cd398cabdfb82a1e3aa188136febe8))
* per-file lineinfo for cross-file inlined device functions ([#599](https://github.com/ccam80/cubie/issues/599)) ([1744635](https://github.com/ccam80/cubie/commit/174463551a5ef77b5fb906b7e12f014269db32e1))
* persistent local scratch array sized from the persistent layout ([#610](https://github.com/ccam80/cubie/issues/610)) ([fc95105](https://github.com/ccam80/cubie/commit/fc951054823c17ca590dd3e232052a3a781929e0))
* step controllers freeze dt and error history on truncated steps ([#596](https://github.com/ccam80/cubie/issues/596)) ([8eb2cd9](https://github.com/ccam80/cubie/commit/8eb2cd93087c0497aa03b993a376c66dbc1695a9))
* step-controller defaults are per-algorithm literature values ([#602](https://github.com/ccam80/cubie/issues/602)) ([38c7654](https://github.com/ccam80/cubie/commit/38c76542500deeccf74673231ac185a238c25567))


### Performance Improvements

* native pooled arrays + async transfers; cut per-solve host overhead ([#618](https://github.com/ccam80/cubie/issues/618)) ([f59650d](https://github.com/ccam80/cubie/commit/f59650d19b24468351c4f8c1e1a5890c2a51f5aa))

## [0.1.1](https://github.com/ccam80/cubie/compare/v0.1.0...v0.1.1) (2026-07-12)


### Bug Fixes

* algorithm steps receive the system's driver count ([#579](https://github.com/ccam80/cubie/issues/579)) ([28a1a0b](https://github.com/ccam80/cubie/commit/28a1a0b5639eef3549a062fd7e7eb139db60ef25))
* as_pandas returns an empty summaries frame with no active metrics ([#584](https://github.com/ccam80/cubie/issues/584)) ([7d9904c](https://github.com/ccam80/cubie/commit/7d9904c67ba0c074392ebe2b617f30db2ddb7652))
* buffer registry releases collected parents ([#595](https://github.com/ccam80/cubie/issues/595)) ([76d81e3](https://github.com/ccam80/cubie/commit/76d81e3989b80096a0e9aaeb37c7b171701a16f9))
* half powers lower to math.sqrt instead of pow ([#586](https://github.com/ccam80/cubie/issues/586)) ([b81fdc1](https://github.com/ccam80/cubie/commit/b81fdc12bf75d7450f4938d8a1f21fac199af3e9))
* hot-swapping algorithm or controller keeps survivors' buffer registrations ([#581](https://github.com/ccam80/cubie/issues/581)) ([cfd4240](https://github.com/ccam80/cubie/commit/cfd4240a6d63f0755c19f74ed629126b20778290))
* memory manager releases collected instances ([#593](https://github.com/ccam80/cubie/issues/593)) ([c898269](https://github.com/ccam80/cubie/commit/c898269a93293c7cd1ad25b4232f37ac1abd51a4))
* repeat chunked solves reuse the group's stored chunk parameters ([#577](https://github.com/ccam80/cubie/issues/577)) ([dc758e3](https://github.com/ccam80/cubie/commit/dc758e325e4f99f6d488f61f99bc6f3f7fdb9dc9))
* set_manual_proportion removes the instance from the auto pool ([#576](https://github.com/ccam80/cubie/issues/576)) ([c688695](https://github.com/ccam80/cubie/commit/c688695b2c1d7dece6d353956430378d53471b7c))
* Solver.update recognises profileCUDA instead of raising KeyError ([#591](https://github.com/ccam80/cubie/issues/591)) ([627a00d](https://github.com/ccam80/cubie/commit/627a00dfa8c8cfcd2b2b5cc02d8fc5afa52c0955)), closes [#590](https://github.com/ccam80/cubie/issues/590)
* underived states convert to observables instead of crashing ([#585](https://github.com/ccam80/cubie/issues/585)) ([adb37c7](https://github.com/ccam80/cubie/commit/adb37c76b81ec1c3e2b6c11c1dcb4e3d72514ddc))
* user device functions resolve inside generated modules ([#578](https://github.com/ccam80/cubie/issues/578)) ([87ca19b](https://github.com/ccam80/cubie/commit/87ca19bc9b1d208abcc07456911420817e7e1c26))

## [0.1.0](https://github.com/ccam80/cubie/compare/v0.0.8...v0.1.0) (2026-07-09)


### Features

* add BiCGSTAB linear solver and Jacobi preconditioner ([#524](https://github.com/ccam80/cubie/issues/524)) ([51e0ed5](https://github.com/ccam80/cubie/commit/51e0ed53f57b60f288d814e2e075037dbbec2a82))
* add Neumann preconditioner convergence diagnostic ([#523](https://github.com/ccam80/cubie/issues/523)) ([950034d](https://github.com/ccam80/cubie/commit/950034db4ceabebd2778358c8c38605cf2cae9be))
* max_registers kwarg caps per-thread registers via cuda.jit ([#553](https://github.com/ccam80/cubie/issues/553)) ([b106b39](https://github.com/ccam80/cubie/commit/b106b39b282e617156e1aadbdef4c980d01cccb3))


### Bug Fixes

* adaptive rejection can no longer loop forever ([#529](https://github.com/ccam80/cubie/issues/529)) ([#535](https://github.com/ccam80/cubie/issues/535)) ([d13fc3a](https://github.com/ccam80/cubie/commit/d13fc3a9c1c7b5a2ffd0ab8a997b1a096cf05934))
* constant power exponents lower to multiplication chains ([#552](https://github.com/ccam80/cubie/issues/552)) ([ff4da5b](https://github.com/ccam80/cubie/commit/ff4da5b230d6ddcb62d00882cb25ceb63553b03c))
* CuPy is the single device memory allocation provider ([#561](https://github.com/ccam80/cubie/issues/561)) ([32a2617](https://github.com/ccam80/cubie/commit/32a26173ea21c7a54d0d61ee92e57579d50ed55c))
* derivative metric history guards gate on the sample counter, not on nonzero values ([#542](https://github.com/ccam80/cubie/issues/542)) ([bd3715b](https://github.com/ccam80/cubie/commit/bd3715b0c83018a75d7cffa2824b2eb2c74e7a6b))
* dict inputs map to state/parameter slots by key name ([#530](https://github.com/ccam80/cubie/issues/530)) ([#532](https://github.com/ccam80/cubie/issues/532)) ([1787c3a](https://github.com/ccam80/cubie/commit/1787c3ad0bd691f0f1be1cf2471980d9f3d32427))
* driver dict inputs map to system driver slots by key name ([#536](https://github.com/ccam80/cubie/issues/536)) ([de05c78](https://github.com/ccam80/cubie/commit/de05c7806268c37ef046a10122e0201e7eda2d26))
* drop no-effect and unreachable code from numba compat shims ([329d6fe](https://github.com/ccam80/cubie/commit/329d6fea50ff58a6d5345f80e2c1ec9984777f93))
* inner-solver tolerances default to controller tolerance over ten ([#538](https://github.com/ccam80/cubie/issues/538)) ([ec6158c](https://github.com/ccam80/cubie/commit/ec6158ce937930e8638704cc9d7c99b366c89a5b))
* negative tolerances rejected; sdirk_2_2 declared errorless ([#549](https://github.com/ccam80/cubie/issues/549)) ([0dc5cf6](https://github.com/ccam80/cubie/commit/0dc5cf64b492915cd3b97e71e77032cb51e33839))
* nested solver shared buffers sized correctly through the allocation chain ([#545](https://github.com/ccam80/cubie/issues/545)) ([ed18917](https://github.com/ccam80/cubie/commit/ed1891764a200b5b409ecee338510c335c5d48b8))
* no-op controller/algorithm updates keep buffer registration ([#525](https://github.com/ccam80/cubie/issues/525)) ([#533](https://github.com/ccam80/cubie/issues/533)) ([c49660e](https://github.com/ccam80/cubie/commit/c49660e3bfdcd166b431a045ef50f644490ae175))
* output schedules tolerate f32 drift and zero-gap boundaries ([#566](https://github.com/ccam80/cubie/issues/566)) ([d8407df](https://github.com/ccam80/cubie/commit/d8407dff38fcb037c54760c47618e8e488de2393))
* package source edits invalidate compiled-kernel and codegen caches ([#544](https://github.com/ccam80/cubie/issues/544)) ([2415657](https://github.com/ccam80/cubie/commit/24156578e84b948681926119cdce62b1fe5140b1))
* PI/PID controller default gains use canonical predictive values ([#537](https://github.com/ccam80/cubie/issues/537)) ([3f4020c](https://github.com/ccam80/cubie/commit/3f4020cabcc187dc23fa79ec826fbeee8b26c2b9))
* rosenbrock23 tableau transformed to the increment convention ([#527](https://github.com/ccam80/cubie/issues/527)) ([#534](https://github.com/ccam80/cubie/issues/534)) ([0e11892](https://github.com/ccam80/cubie/commit/0e11892050103ee9797741faa273cf38915acbcf))
* summary legends populate for summary-only solves; fused metrics report requested names ([#559](https://github.com/ccam80/cubie/issues/559)) ([762dd5e](https://github.com/ccam80/cubie/commit/762dd5e2abfbbd871f1de4fe9d410b8312f16dc6))
* transient step failures no longer stain the run status word ([#539](https://github.com/ccam80/cubie/issues/539)) ([d6c41dc](https://github.com/ccam80/cubie/commit/d6c41dc1cc86755af8a24a404586debe4f3ddb0d))
* unconsumed Solver kwargs raise; save_last fires on exact t_end landing ([#540](https://github.com/ccam80/cubie/issues/540)) ([46e0d57](https://github.com/ccam80/cubie/commit/46e0d570f6bcc20758faf7801b984d3f4b2bd197))


### Documentation

* solve() configuration reference and per-algorithm defaults ([#560](https://github.com/ccam80/cubie/issues/560)) ([1248ea3](https://github.com/ccam80/cubie/commit/1248ea363468148383397b39f97d1c9cc36f2e30))
* three runnable tutorials cover sweeps, summaries, stiff solving ([#551](https://github.com/ccam80/cubie/issues/551)) ([540d7a5](https://github.com/ccam80/cubie/commit/540d7a5a9214558722c22ed13beae069075ac644))
* user guide claims match current behaviour ([#550](https://github.com/ccam80/cubie/issues/550)) ([31fbae6](https://github.com/ccam80/cubie/commit/31fbae647806eec070042c1e6ca25505ca5e4f52))

## [0.0.8](https://github.com/ccam80/cubie/compare/v0.0.7...v0.0.8) (2026-07-06)


### Features

* CellML loader removes GHK singularities by default ([#522](https://github.com/ccam80/cubie/issues/522)) ([ef99849](https://github.com/ccam80/cubie/commit/ef9984988187b9478bda52024e3c261b23f995cf))
* cellmlmanip vendored under cubie.vendored to lift the Pint&lt;0.20 pin ([f30f9f3](https://github.com/ccam80/cubie/commit/f30f9f3f64ed0ed6547a3da02bfb40ab8f924ab0))
* central CUBIE_RESULT_CODES status vocabulary ([72a7a36](https://github.com/ccam80/cubie/commit/72a7a36c7f0838822501f2c0a6e9a1a28dbad4b6))
* constants/params/inits setting gui added (for large models) ([e6af22b](https://github.com/ccam80/cubie/commit/e6af22b380786297a6ae3cd6f1836e620f16fd00))
* more-sensible defaults are set when only a subset of dt_min, dt_max, dt are set ([f552dda](https://github.com/ccam80/cubie/commit/f552dda1c928f17bc88e7efc24492b16eff6cd60))
* numba-cuda compile-time patches ship as a compat module ([dc9fe8f](https://github.com/ccam80/cubie/commit/dc9fe8fa1c4e2b68f10a786b75635adc84f76bb2))


### Bug Fixes

* adaptive loops no longer hang when newton solver fails on a save boundary ([#519](https://github.com/ccam80/cubie/issues/519)) ([f552dda](https://github.com/ccam80/cubie/commit/f552dda1c928f17bc88e7efc24492b16eff6cd60))
* broken ODEData beta/gamma properties removed and the mass matrix folded into the compile-cache key ([b5ad20c](https://github.com/ccam80/cubie/commit/b5ad20cb602e78308e6e361553f4b3beeaa2ce62))
* CUBIECache initialises numba-cuda launch-config cache state ([9bb8e39](https://github.com/ccam80/cubie/commit/9bb8e39fd210d26d26808b0acd2b95c397d3691c))
* dynamic-shared block-size reduction floors at one warp ([7d2b5fb](https://github.com/ccam80/cubie/commit/7d2b5fb956f3608719080d7e7b713951b02b37ea))
* power-expansion rewrite parenthesizes x*x, preserving division precedence ([503c76c](https://github.com/ccam80/cubie/commit/503c76c168c69072afd28d1251deb12392e94cab))
* pyfunc now doesn't relabel output states ([45b94c6](https://github.com/ccam80/cubie/commit/45b94c6ad9b4ad992233321bedfe39df46be944c))
* sub-warp block sizes reserved for hardware necessity ([c51af85](https://github.com/ccam80/cubie/commit/c51af85ec9d738e3a526f1f210475ecbbe4af2fa))
* tableau weight-sum validation is two-sided and RK-scoped ([94d8d2a](https://github.com/ccam80/cubie/commit/94d8d2a3fcef334d893967ea008e3a703bb78f04))
* update event handle handling to inspect the value of handle directly. ([5926b8c](https://github.com/ccam80/cubie/commit/5926b8c0afa36ca41bde01b63ddbc9fc00955821))


### Documentation

* add explanation of output timing, loop duration/start timing, and step timing to the user guide ([f552dda](https://github.com/ccam80/cubie/commit/f552dda1c928f17bc88e7efc24492b16eff6cd60))
* chaste_codegen attribution removed from the Jacobian generator, which shares no provenance with it ([259278a](https://github.com/ccam80/cubie/commit/259278a4f13e05de1874ed40c574e302412c9dca))
* testing guidance restored to the mandatory centralised-fixture directive and the phantom is_device/CUDA-availability rule removed ([29215fb](https://github.com/ccam80/cubie/commit/29215fb2a19f802356506a02b364c55ce2ea2e97))
* testing rules hardened so mocks require an explicit user exception and tests are never softened with lax assertions ([b9099f6](https://github.com/ccam80/cubie/commit/b9099f62264fb9308b1a313a94e8c2060ebbc3bc))
* top-level AGENTS.md written, root CLAUDE.md symlink added, and GitHub agent-instruction files removed ([87ba31f](https://github.com/ccam80/cubie/commit/87ba31f3059c625e55ed88569b83a5ea066a06e2))


### Miscellaneous Chores

* release 0.0.8 ([b433b67](https://github.com/ccam80/cubie/commit/b433b67f7348651766d2a807582ecaf3947adf55))

## [0.0.7](https://github.com/ccam80/cubie/compare/v0.0.6...v0.0.7) (2026-01-20)


### Features

* add runtime logging infrastructure for GPU kernels and memory transfers ([#289](https://github.com/ccam80/cubie/issues/289)) ([431425d](https://github.com/ccam80/cubie/commit/431425d5026e90beaf7367bce1863c7f61c2b34b))
* add unified save_variables and summarise_variables parameters to solver interface ([#342](https://github.com/ccam80/cubie/issues/342)) ([c7d7531](https://github.com/ccam80/cubie/commit/c7d75317f9525da4f4e2f1f876676bc6fbd54669))
* cellml-generated systems now cached ([#510](https://github.com/ccam80/cubie/issues/510)) ([92b21e0](https://github.com/ccam80/cubie/commit/92b21e01facfa4a5a23b73f8a777efaefae4cdd7))
* enable driver interpolator profiling in all_in_one.py ([#419](https://github.com/ccam80/cubie/issues/419)) ([1574ec9](https://github.com/ccam80/cubie/commit/1574ec9b548c8f4690f9ac92e927d69bee2ae571))
* File-based caching implemented  ([#491](https://github.com/ccam80/cubie/issues/491)) ([1bfe68b](https://github.com/ccam80/cubie/commit/1bfe68b2323ddd7dd48b1d3c51dc969b988f5832))
* MultipleInstanceCUDAFactory subclass (and matching config) now handle cases like newton_atol and krylov_atol when instantiating multiple of the same base class ([d674bcd](https://github.com/ccam80/cubie/commit/d674bcd95d9e1757038b50cb386151591bf74787))
* scaled norm function now available as a CUDAFactory for repeated use ([d674bcd](https://github.com/ccam80/cubie/commit/d674bcd95d9e1757038b50cb386151591bf74787))
* Scaled tolerance in Newton-Krylov solver ([#473](https://github.com/ccam80/cubie/issues/473)) ([d674bcd](https://github.com/ccam80/cubie/commit/d674bcd95d9e1757038b50cb386151591bf74787))
* Solve functions now save the final value on loop exit when no timing parameters are given. ([d9fb64b](https://github.com/ccam80/cubie/commit/d9fb64bd19d573c32b26d95323392d33f5c62d39))
* Time-domain save settings now decoupled from summary metric settings ([d9fb64b](https://github.com/ccam80/cubie/commit/d9fb64bd19d573c32b26d95323392d33f5c62d39))


### Bug Fixes

* build_grid now takes None parameters/initial conditions in solver.py ([be4bab9](https://github.com/ccam80/cubie/commit/be4bab9729341ca980be1321f4d850a2020b9dea))
* chunking fails when VRAM is limited due to stride incompatibility ([#487](https://github.com/ccam80/cubie/issues/487)) ([b8a486e](https://github.com/ccam80/cubie/commit/b8a486e36a88f82ba8ee089282f7c84c09d7a498)), closes [#438](https://github.com/ccam80/cubie/issues/438)
* codegen hashing now session-independent (and so working... this time) ([59bb488](https://github.com/ccam80/cubie/commit/59bb488617ec00388d85017fd76221a04019dc76))
* correct false circular dependency error in topological_sort ([#422](https://github.com/ccam80/cubie/issues/422)) ([97c13be](https://github.com/ccam80/cubie/commit/97c13bec7dd31c9cfedba6503783c19ee9a9c59a))
* default neumann preconditioner order set to 2 ([2434358](https://github.com/ccam80/cubie/commit/24343589fa30f28da630517bee26b691bbf48c2a))
* DIRK codegen pipeline now decoupled from rosenbrock cache planning. ([80171b5](https://github.com/ccam80/cubie/commit/80171b52a33290c2f0f1ae10be8eed5e95f8308b))
* dummy-kernel based compile time logging removed (it doubled compile time) ([431425d](https://github.com/ccam80/cubie/commit/431425d5026e90beaf7367bce1863c7f61c2b34b))
* Internal code generation variables prefixed to avoid name clashes ([#466](https://github.com/ccam80/cubie/issues/466)) ([90e8ca3](https://github.com/ccam80/cubie/commit/90e8ca31070193a1db24c584af424d5ab11c0b20)), closes [#373](https://github.com/ccam80/cubie/issues/373)
* load_cellml_model surfaced to toplevel import ([54b05e0](https://github.com/ccam80/cubie/commit/54b05e09b7e0bd961cf76e931c98470b3f3aa33f))
* loop now exits on irrecoverable-error status codes ([ba800a0](https://github.com/ccam80/cubie/commit/ba800a0df6142ae082f39d01fb57864fc0c5990b))
* map CellML time variable to standard 't' symbol ([#425](https://github.com/ccam80/cubie/issues/425)) ([261c109](https://github.com/ccam80/cubie/commit/261c1092ba9a50b703a1bc949957217462bafd6f))
* Newton-krylov solver no longer propagates krylov non-convergence or max_backtrack errors if it recovers ([1411135](https://github.com/ccam80/cubie/commit/1411135b70268377f23df251e7b13de518803499))
* Parsed system definition now hashed properly so generated code is properly cached ([655e54a](https://github.com/ccam80/cubie/commit/655e54af6b7e9ad86559c36e447fcf3c04acfd74))
* patch event.query() bug in numba by swapping handle ([d43ae8f](https://github.com/ccam80/cubie/commit/d43ae8f8d441353aa56815ff31db9ea06bf9094e))
* Refactor BatchGridBuilder -&gt; BatchInputHandler ([#437](https://github.com/ccam80/cubie/issues/437)) ([ffb8478](https://github.com/ccam80/cubie/commit/ffb8478a4bfc01bdba1d3cc146016b0cd2bf723e))
* repeated CSE calls no longer raise warning about an already-used CSE symbol. ([16785f3](https://github.com/ccam80/cubie/commit/16785f3a4641b719c2a70b11becda5533f6e4e2e))
* Rosenbrock step config's hash now deterministic for caching ([105befb](https://github.com/ccam80/cubie/commit/105befbbef2419e11e92fe781b3f2853b7267081))
* Runs with impossible time settings now raise sensible errors ([#465](https://github.com/ccam80/cubie/issues/465)) ([d46a7b2](https://github.com/ccam80/cubie/commit/d46a7b2fea3eb4d63753e1679f0a95114fc12022)), closes [#440](https://github.com/ccam80/cubie/issues/440)
* skip codegen on cache hit in get_solver_helper ([#512](https://github.com/ccam80/cubie/issues/512)) ([cfc6b0b](https://github.com/ccam80/cubie/commit/cfc6b0b826659b1e5a132a017f5d1236fcf87cbc))
* state-aware derivative detection to avoid misinterpreting d-prefixed auxiliaries ([#468](https://github.com/ccam80/cubie/issues/468)) ([fee6d70](https://github.com/ccam80/cubie/commit/fee6d70f4612ef6a8cb0d52da1a0ef6627516066))
* Stride incompatibility fixed when array is sliced to fit in VRAM ([b8a486e](https://github.com/ccam80/cubie/commit/b8a486e36a88f82ba8ee089282f7c84c09d7a498))
* timelogger printouts now include cache hit/miss messaging ([#497](https://github.com/ccam80/cubie/issues/497)) ([6323746](https://github.com/ccam80/cubie/commit/6323746cad81a530514f02f9c30ed43fcc147597))


### Performance Improvements

* Convert whole-module imports to explicit imports in CUDAFactory files ([#443](https://github.com/ccam80/cubie/issues/443)) ([9cebe45](https://github.com/ccam80/cubie/commit/9cebe45b93a9b5d5f189dee971edf36ecbb9431c))


### Miscellaneous Chores

* release 0.0.7 ([40a0d75](https://github.com/ccam80/cubie/commit/40a0d750599fd7e58c1dd7cb905d01e1264b273b))

## [0.0.6](https://github.com/ccam80/cubie/compare/v0.0.5...v0.0.6) (2025-12-27)


### Features

* ``raw`` output type added to output device array copies with no processing ([22cef9f](https://github.com/ccam80/cubie/commit/22cef9ff02000d1e60a6576a3ef54a54918e6658))
* add time logging to cellml import ([#257](https://github.com/ccam80/cubie/issues/257)) ([6a220f8](https://github.com/ccam80/cubie/commit/6a220f8e78653f1727967b2092002f78ba41db71))
* Additional summary output metrics added ([#212](https://github.com/ccam80/cubie/issues/212)) ([daccbae](https://github.com/ccam80/cubie/commit/daccbae24945c081d75585a6052ec95a45885808))
* Buffer indexing, sizing, and locating now consolidated into a BufferSettings object ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* Buffer memory locations on GPU now user-selectable (between local and shared) ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* build_grid() surfaced to user API ([#338](https://github.com/ccam80/cubie/issues/338)) ([45cfa90](https://github.com/ccam80/cubie/commit/45cfa90a77f2892d29966bd856215c4b54475cc0))
* CellML to Cubie adapter layer added ([#221](https://github.com/ccam80/cubie/issues/221)) ([b8f448e](https://github.com/ccam80/cubie/commit/b8f448e532c6c5410fd06aa13a87531301468de3))
* CUDAFactory.update() now updates nested dicts and attrs classes ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* device buffers are now togglable (local, shared, local persistent) and managed centrally by buffer_registry ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* Global stores now have cache write-through hints, closes [#291](https://github.com/ccam80/cubie/issues/291) ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* load_from_cellml updated to parse complicated models ([#238](https://github.com/ccam80/cubie/issues/238)) ([50341f6](https://github.com/ccam80/cubie/commit/50341f62704f44f95f9e591635609e3c1e3bcd74))
* newton and linear solvers now fully-functional CUDAFactory subclasses ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* optional configuration parameters are no longer explicit in inits, they are filtered and collected in kwargs ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* py39 compatibility removed due to a Numba update. ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* Solver API now skips extra memory and grid-building work when possible ([#324](https://github.com/ccam80/cubie/issues/324)) ([5f225a0](https://github.com/ccam80/cubie/commit/5f225a023fc38d17e90a688f51a91d5b7c9db6ee))
* summary metrics combined (eg. extrema, [mean, std, rms]) to reduce buffer space ([daccbae](https://github.com/ccam80/cubie/commit/daccbae24945c081d75585a6052ec95a45885808))
* summary metrics now respect numerical precision ([daccbae](https://github.com/ccam80/cubie/commit/daccbae24945c081d75585a6052ec95a45885808))
* Sympy inputs from CellML now go Sympy-&gt;Sympy instead of through strings ([#259](https://github.com/ccam80/cubie/issues/259)) ([23e201d](https://github.com/ccam80/cubie/commit/23e201df19c0c607ee8bc4512f2b04147bdb29be))
* time logging added to parsing, codegen, and CUDA compilation ([#256](https://github.com/ccam80/cubie/issues/256)) ([5d8f75b](https://github.com/ccam80/cubie/commit/5d8f75b800ef59f02a049e037ac9333f03fdbefd))
* Trajectories with errors (nonzero status codes) now return NaNs in solveresult ([#333](https://github.com/ccam80/cubie/issues/333)) ([6068ebc](https://github.com/ccam80/cubie/commit/6068ebc68e0af517462744e8ead23eb57c8c578a))
* update() methods now unpack settings dicts provided to them ([#332](https://github.com/ccam80/cubie/issues/332)) ([f016e4d](https://github.com/ccam80/cubie/commit/f016e4dfab6c8b22b759f363f68ad83380aba940))
* Warp-friendly FSAL caching implented, redundant accumulation removed ([#211](https://github.com/ccam80/cubie/issues/211)) ([96a9dd0](https://github.com/ccam80/cubie/commit/96a9dd00fc35e9d65fede8ade7f6579ec3e896e3))


### Bug Fixes

* _ensure_context() method added to avoid segfaults in CI ([#265](https://github.com/ccam80/cubie/issues/265)) ([60acecc](https://github.com/ccam80/cubie/commit/60acecccb787913c0837d072f40d9e51e347fd3a))
* `shift` value in standard deviation calcs now updates after each save. ([073d406](https://github.com/ccam80/cubie/commit/073d406f4578cff67db5d223da607e5ecd437138))
* Adaptive step controllers now sum errors correctly, closes [#302](https://github.com/ccam80/cubie/issues/302) ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* add set_stride_order method for access from solver ([e4b80d3](https://github.com/ccam80/cubie/commit/e4b80d3985366c9b1f2ce2f5da7baac3c4210452))
* All algorithms now exit when the next save time is &gt; end time ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* buffer aliasing logic now child-location-agnostic ([377ad10](https://github.com/ccam80/cubie/commit/377ad10c4818d4faeb9c1e8cb6ecbbb3a871ead5))
* Compile-timing kernels now use device arrays ([16bf1d6](https://github.com/ccam80/cubie/commit/16bf1d62953596610e628d2677cc23ef5c6773d8))
* consolidate timing parameter sets for fewer fixture builds ([#350](https://github.com/ccam80/cubie/issues/350)) ([4698e9e](https://github.com/ccam80/cubie/commit/4698e9e78bbba37c0a9a28e06832b160c11efad0))
* contiguous arrays now marked as such for the compiler to do its grim work ([#276](https://github.com/ccam80/cubie/issues/276)) ([fe97291](https://github.com/ccam80/cubie/commit/fe972918a79518dc407750a7be552182291701ed))
* Controller-algorithm compatibility enforced ([4ab0230](https://github.com/ccam80/cubie/commit/4ab0230c9261e3857f099167db2735db6a6c2955))
* correct FIRK algorithm implementation ([#408](https://github.com/ccam80/cubie/issues/408)) ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* correct fsal warp-vote implementation ([60448f1](https://github.com/ccam80/cubie/commit/60448f10b3976533152450fa3d2b56ebae0af337))
* Counters array added to ERK signature ([99db833](https://github.com/ccam80/cubie/commit/99db8337e75be96ed69fe582b230264e1e2ea425))
* CUDA-breaking simsafe 'local' module adapter removed ([d205979](https://github.com/ccam80/cubie/commit/d205979d9894057a1787ac4b0307eca8ea21ee9c))
* CUDAFactories now only precompile if timelogging is on ([836da2a](https://github.com/ccam80/cubie/commit/836da2a417d197bbfdc06ad8188b991ae184ca78))
* dead code and duplications pruned from codegenned device functions ([#266](https://github.com/ccam80/cubie/issues/266)) ([de0dda8](https://github.com/ccam80/cubie/commit/de0dda8f14a4c50ec1c1794c01634907bd87d4c7))
* default precision types have been removed from all but entry points ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* fast path through batchgridbuilder for arrays provided verbatim added ([080ab66](https://github.com/ccam80/cubie/commit/080ab66d7fc8b4a6cd9e4291392e773c44152175))
* FIRK no longer does an extra f(x) calculation after performing its nonlinear solve ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* Fixed steppers now use a slower but more sensible save time incrementing technique ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* FSAL warp test now doesn't break everything ([2e68692](https://github.com/ccam80/cubie/commit/2e68692da71f643aa725ed0a1ad8c10a8c2b980a))
* gustafsson controller prev error ratio flipped, various test edits for GPU runs ([3e04438](https://github.com/ccam80/cubie/commit/3e04438881a55ba03306b6f112e35c2d3e3400b5))
* host arrays now pinned to facilitate asynchronous transfers ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* infinite loops in adaptive steppers under dummy compile now finite ([b0547fd](https://github.com/ccam80/cubie/commit/b0547fdcc0c379c84af10375ceedf479da67dc4d))
* lineinfo toggle for CUDA compilation now conditional on CUDASIM status ([#280](https://github.com/ccam80/cubie/issues/280)) ([6d5bf57](https://github.com/ccam80/cubie/commit/6d5bf57a6a381379e9e22f9c5901326051dd5835))
* Loop iterators now 32-bit ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* loops now accumulate time in f64, no longer get stuck when dt &lt; 1e-7 * time ([#281](https://github.com/ccam80/cubie/issues/281)) ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb)), closes [#272](https://github.com/ccam80/cubie/issues/272)
* make shape validator np.int compatible in BaseArrayManager.py ([e77d800](https://github.com/ccam80/cubie/commit/e77d8008de8c1cecd715df6c29bf79794c10bd6b))
* make stage_increment buffer persistent in rosenbrock  ([#411](https://github.com/ccam80/cubie/issues/411)) ([4b0b6bc](https://github.com/ccam80/cubie/commit/4b0b6bc2102bf7af5f4bee2877a0bce5c0b5a860))
* Missing iteration_counters type added to device signature in BatchSolverKernel ([6eefd67](https://github.com/ccam80/cubie/commit/6eefd67140767d8b452b99fbdcf8d19bf7e91edb))
* move stream sync function to after chunked queue ([e4b80d3](https://github.com/ccam80/cubie/commit/e4b80d3985366c9b1f2ce2f5da7baac3c4210452))
* n_saves calculation now includes start time, closes [#282](https://github.com/ccam80/cubie/issues/282) ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* numeric literals now wrapped with precision() or int32 in CUDA code generation ([#258](https://github.com/ccam80/cubie/issues/258)) ([21850f2](https://github.com/ccam80/cubie/commit/21850f293b6743e2fe677b4fccc85129f86cdbca))
* partially update cuda signatures with contiguity ([8754661](https://github.com/ccam80/cubie/commit/87546613825922befad0c2145e28807b04ab62d8))
* reduced nonlinear solver memory footprint (3n → 2n buffers) ([#224](https://github.com/ccam80/cubie/issues/224)) ([c4d95d4](https://github.com/ccam80/cubie/commit/c4d95d419f72ce12e1242767fd9bd7056e81ce5d))
* Remaining hard-coded buffers now managed by buffer_registry ([#412](https://github.com/ccam80/cubie/issues/412)) ([92e9313](https://github.com/ccam80/cubie/commit/92e93139a54c6e4e290bc8a6b4e0de85d2890a24))
* remove redundant overwrite of initial values host array until [#76](https://github.com/ccam80/cubie/issues/76) is implemented in device code ([bc39fbb](https://github.com/ccam80/cubie/commit/bc39fbbc52ea3475cbe2b8581a59f8ae6c7fdfc9))
* Replace PEP 604 union syntax with Union[] for Python 3.8 compatibility ([#236](https://github.com/ccam80/cubie/issues/236)) ([ea1e1fb](https://github.com/ccam80/cubie/commit/ea1e1fb89db14e798d0d12a1ed2e68f6b114c81a))
* shared and local memory requirement names now consistent across package. ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* size-1 runs no longer break Cubie (closes [#142](https://github.com/ccam80/cubie/issues/142)) ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* steps now own solvers, and newton_krylov solver now owns it's internal krylov solver ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* Tableau-driven algorithm loop indexing reorganised to give the compiler an easier job ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* tsit5 tableau corrected ([480df1a](https://github.com/ccam80/cubie/commit/480df1aed16fee4be48781315afddb44bfbfefeb))
* types corrected in all step algorithms and solvers to run in 'precision' only ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* update all-in-one rosenbrock function to use updated prod format ([fa32a16](https://github.com/ccam80/cubie/commit/fa32a1607824a2342dcb46649b7f1d3ca8cbd86f))
* update placeholder parameters to not hide exception ([8754661](https://github.com/ccam80/cubie/commit/87546613825922befad0c2145e28807b04ab62d8))
* updating a solverkernel with new outputtypes now correctly changes buffer and output array sizes. ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* Use a warp-vote for FSAL caching, otherwise there is no benefit and potential divergence ([d0334a2](https://github.com/ccam80/cubie/commit/d0334a2841518467f2b794fa7a325861b7e9471a))
* Vern7 tableau corrected to match source ([e6b900a](https://github.com/ccam80/cubie/commit/e6b900a0452926a953709e1cf65f89fc147a1c97))


### Performance Improvements

* DIRK function profiled and streamlined ([#355](https://github.com/ccam80/cubie/issues/355)) ([fd1f639](https://github.com/ccam80/cubie/commit/fd1f639b1394ce8d1f6068d5cc846daea560a2d4))
* firk rewirked for a large speedup ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* newton and linear solver branching and predication removed for a reasonable speedup ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))
* rosenbrock function reworked for a moderate speedup ([39107a9](https://github.com/ccam80/cubie/commit/39107a925d63e4f708b3e18ef198cb75ead52262))


### Documentation

* cubie_internal_structure.md filled with agent reference ([#177](https://github.com/ccam80/cubie/issues/177)) ([02738e0](https://github.com/ccam80/cubie/commit/02738e06f7343910e292721d43cdc0ec127adf4e))


### Miscellaneous Chores

* Release 0.0.6 ([baa2523](https://github.com/ccam80/cubie/commit/baa2523e21c4278dd2c88b1268b689f6ef8a6bac))



## [0.0.5](https://github.com/ccam80/cubie/compare/v0.0.4...v0.0.5) (2025-11-04)


### Features

* "Instrumented" device steps added for diagnostics ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* additional ERK, DIRK, Rosenbrock tableaus added ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* Algorithms now set their own step control defaults ([#139](https://github.com/ccam80/cubie/issues/139)) ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Array managers now support heterogeneous arrays within the same container ([0905422](https://github.com/ccam80/cubie/commit/0905422b8fbaf6c1b05f127d162a2f44bc40c53a))
* Compile-settings updates now won't force a rebuild if the value hasn't changed. ([5a9e281](https://github.com/ccam80/cubie/commit/5a9e281bb16455c37ca70c85d56831d50f809fc7))
* DIRK and DIRKTableaus added ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Explicit and Diagonally-Implicit Runge Kutta algorithms added ([#151](https://github.com/ccam80/cubie/issues/151)) ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Explicit RK and Tableau added, closes [#83](https://github.com/ccam80/cubie/issues/83) ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Fully-Implicit Runge-Kutta (FIRK) Methods implemented ([#162](https://github.com/ccam80/cubie/issues/162)) ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* Generic Butcher Tableau implemented ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* Generic Rosenbrock-W methods added ([#148](https://github.com/ccam80/cubie/issues/148)) ([6abfed2](https://github.com/ccam80/cubie/commit/6abfed2c0d051b597b11ea3601a4806eecfc7aac))
* minimal FSAL caching added to DIRK, ERK, Rosenbrock ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* N-stage flattened linear operators, preconditioners, nonlinear residual codegen added ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* Parser now processes indexed arrays as variables ([#152](https://github.com/ccam80/cubie/issues/152)) ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* rodasnp methods, dop853, tsit5, vern7 ([87b53bc](https://github.com/ccam80/cubie/commit/87b53bc6b87227db223989a5576e17846448b685))
* Rosenbrock methods now 100% more rosenbrock ([#157](https://github.com/ccam80/cubie/issues/157)) ([35641e6](https://github.com/ccam80/cubie/commit/35641e6ab24f76193a03e29c01c64822298dba63))
* status codes now aggregated by batchSolverKernel ([0905422](https://github.com/ccam80/cubie/commit/0905422b8fbaf6c1b05f127d162a2f44bc40c53a))
* Steps and step controllers now have a unified argument-filtering factory ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Tableau libraries and tableau resolvers/getters added ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* There are now auxiliary-cached jacobian functions for reusing some computational work. ([6abfed2](https://github.com/ccam80/cubie/commit/6abfed2c0d051b597b11ea3601a4806eecfc7aac))
* Third and fourth order SDIRK tableaus added ([91ee1e9](https://github.com/ccam80/cubie/commit/91ee1e967d343b38ed521f52012ca4a414ecdbd2))
* Time-derivative helpers added for symbolic functions and interpolated arrays ([35641e6](https://github.com/ccam80/cubie/commit/35641e6ab24f76193a03e29c01c64822298dba63))
* Very rough caching of jvp nodes implemented for rosenbrock solvers. ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* working arrays and quantities in algorithms and solvers now draw from a memory "pool", allowing easier reuse ([b0f5b6f](https://github.com/ccam80/cubie/commit/b0f5b6f879d82caa700c853042a860736da02714))


### Bug Fixes

* add error to sdirk 4, correct controllers for loop tests ([47a6cb1](https://github.com/ccam80/cubie/commit/47a6cb1fb20a53b55c38711668abd3b98fa33ba5))
* Added (non-CI) testing for DIRK loops added. ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* batchgridbuilder class now has static wrappers for batchgridbuilder helper functions. ([74c08fa](https://github.com/ccam80/cubie/commit/74c08fa0a9846c988399e9893a1acee3193a62ca))
* BatchsolverKernel's compile settings now features the compile-critical settings it always should have had. ([f383157](https://github.com/ccam80/cubie/commit/f383157a06180841071bf9df90d23c1081c907f2))
* Buffer footprints reduced by aliasing vectors with disjoint lifetimes ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* correct off-by-one error and datatype discrepancy in cpu driver evaluator ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* DIRK now accumulates rhs's and scales after stages, reducing round-off ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))
* faulty implementation of ode23s removed, dop853 tableau amended ([d5a784c](https://github.com/ccam80/cubie/commit/d5a784c366a22d81b4c12651b0536c231263098d))
* matplotlib now spelled correctly in pyproject.toml ([3773ff8](https://github.com/ccam80/cubie/commit/3773ff8c0a73fca8986794d82289f2d0e3a3c169))
* Meaty loop tests confined to test_ode_loop.py ([7fe5677](https://github.com/ccam80/cubie/commit/7fe5677be087bff94a75e1c3e9843938f05c139a))
* numerous numerical errors amended after instrumenting steps ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* precision type hints now correct; no more pesky yellow lines. ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Rosenbrock buffer footprint reduced by 2n ([ed866d7](https://github.com/ccam80/cubie/commit/ed866d7153e4255136a8155f55cb92826f319d6b))
* settings passing from solver now less cramped and hopefully more robust ([2a0efed](https://github.com/ccam80/cubie/commit/2a0efedd7a02a10be5179f0ef84fe72ebfddba84))
* Step controllers no longer mutate error vector ([7fe800b](https://github.com/ccam80/cubie/commit/7fe800b8328f28b26fab1492c9f0c30e63670553))


### Documentation

* Add autodocs subpages for manual project structure docs ([c2c0d7c](https://github.com/ccam80/cubie/commit/c2c0d7cd0c9173cccd38d84d3969d6e66406e4b4))
* add copilot instructions ([#161](https://github.com/ccam80/cubie/issues/161)) ([2bfe6f2](https://github.com/ccam80/cubie/commit/2bfe6f2a28e9e20ed48e855435901ea8145c5a78))
* added "buffer map" comments to generic algorithms to demystify aliasing ([472f960](https://github.com/ccam80/cubie/commit/472f960ae7492520eb1b9c7d22a55f537afae753))
* autodocs param lists now format one line per param ([cb3d74b](https://github.com/ccam80/cubie/commit/cb3d74bf4a578705abf31c797e0662aea3202879))
* Batchsolving module source files now better documented in numpydocs format ([97a0ab8](https://github.com/ccam80/cubie/commit/97a0ab8a9510ea1e46f1aa6cc3a16f0a51ee9ae8))
* de-computer some language in api reference, rejig indexes ([b558994](https://github.com/ccam80/cubie/commit/b5589943dbec791ee41b71c012b148e856352e83))
* increase index depth, force one-param-per-line printing ([b822f4c](https://github.com/ccam80/cubie/commit/b822f4c402f3321c6a399163e5e0497433493a8e))
* more docs organisation ([5631989](https://github.com/ccam80/cubie/commit/563198977c99f688549152feb798335a5eb3bf36))
* more docs organisation ([cd98043](https://github.com/ccam80/cubie/commit/cd98043d02c663ddfe2f3df981cd218ddbc99c95))
* refactor api structure; remove autosummary, implement manual docs ([881ce4a](https://github.com/ccam80/cubie/commit/881ce4ac81c8405f0a8cffe00788de7c987a3dce))
* Refs in "getting started" are now actually refs not highlighted garbage. ([6abfed2](https://github.com/ccam80/cubie/commit/6abfed2c0d051b597b11ea3601a4806eecfc7aac))
* top-level batchsolving docs added ([6cfddfa](https://github.com/ccam80/cubie/commit/6cfddfa8d86d656be4b79b34d068290680e60046))


### Miscellaneous Chores

* release 0.0.5 ([93a7255](https://github.com/ccam80/cubie/commit/93a72554cf461daf58da86b5bbc46bebb197d4d9))

## [0.0.4](https://github.com/ccam80/cubie/compare/v0.0.3...v0.0.4) (2025-10-05)


### Features

* Adaptive step-size controllers added : i (traditional), pi, pid, gustafsson acceleration ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Adaptive time-step controllers now have a programmable dead-band. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* AGENTS.md extended and partially updated to summarise entire project for ai agents ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* arbitrary drivers can now be looped or clamped to zero (smoothly). ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* Backwards Euler implicit fixed-step method added (with and without predictor-corrector mechanism), closes [#114](https://github.com/ccam80/cubie/issues/114). ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Codegen for residual functions, jvps, and various solver helper functions created ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Crank-Nicolson trapezoidal adaptive-step algorithm implemented. ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* cuda simulation patches consolidated for cuda-free environment tests. ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Forcing (driver) terms now adaptive-step friendly ([#132](https://github.com/ccam80/cubie/issues/132)) ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* matrix free solvers added ([1dffd94](https://github.com/ccam80/cubie/commit/1dffd94ae965a72c3f42ea2be09761cedcd10582))
* Nonlinear Newton-Krylov iterative solver with preconditiong added for implicit methods, closes [#101](https://github.com/ccam80/cubie/issues/101), [#102](https://github.com/ccam80/cubie/issues/102), [#111](https://github.com/ccam80/cubie/issues/111) ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* plotting added to driver interpolator - keep an eye on what the machine is doing. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* shared memory padding now closer to optimal (nothing to be done about 64-bit values), closes [#86](https://github.com/ccam80/cubie/issues/86). ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))


### Bug Fixes

* Array "chunking" logic now respects "unchunkable" arrays in allocation ([53141df](https://github.com/ccam80/cubie/commit/53141dfa72a7568813f85c86ef5d9b6f682db856))
* CPU test step controllers now raise dt_too_small errors ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))
* Crank-Nicolson and adaptive controllers now 50% more idiot-error free. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* CUDAFactory now updates underscored config variables as intended ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Many edits to precision settings and flow throughout system (making it work) ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* Observables calculation now occurs in sync with state for adaptive-step loops ([#130](https://github.com/ccam80/cubie/issues/130)) ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))
* pypi version tag trigger now using correct syntax ([3b3caa1](https://github.com/ccam80/cubie/commit/3b3caa1d1e434edc61dc86174d211ac83444383f))
* some sign confusion and bogus gains corrected in step controllers ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))


### Documentation

* Batch arrays re-docstringed in keeping with rest of library. ([b0fafd9](https://github.com/ccam80/cubie/commit/b0fafd9f4c9f269edf5ae003fe64cd9309d2b43b))
* fix circular import for docs building ([64d8933](https://github.com/ccam80/cubie/commit/64d8933e965b52f80664f2f5ddefc5ab03d96077))
* manual docs added for integrators, memory modules. submodules of systems, outputhandling  documented. ([1d903f2](https://github.com/ccam80/cubie/commit/1d903f2bfff3c3732086c18efa4a66660524bf29))
* step controller comparison added to docs/examples ([19af8ff](https://github.com/ccam80/cubie/commit/19af8ff620419155e4c6e8d3512aa1224421d3ad))
* top-level summaries of odesystems and outputhandling ([a9cb4c0](https://github.com/ccam80/cubie/commit/a9cb4c08383aa452d64a5cf63bec9b3cb86d5b75))
* update docstrings in outputhandling and odesystems root directorys ([87bb133](https://github.com/ccam80/cubie/commit/87bb133643a8b8b14ff6731469e2ef5e7eded3ee))


### Miscellaneous Chores

* release 0.04 ([ed5045a](https://github.com/ccam80/cubie/commit/ed5045ad6e5677cc1f4543d0c1a8673c76028fe2))

## [0.0.3](https://github.com/ccam80/cubie/compare/v0.0.2...v0.0.3) (2025-09-04)


### Features

* Parser accepts and translates sympy and user-provided functions ([#108](https://github.com/ccam80/cubie/issues/108)) ([25f3c5d](https://github.com/ccam80/cubie/commit/25f3c5d89edca3d85040b1438995d45f010b99a4))
* Symbolic input parsing added ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))
* symbolic interface and analytical jacobian generation added ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))


### Bug Fixes

* BaseArrayManager.py once again contains all of it's methods, after half of the class was sausage-fingered clean off. ([96ba1cb](https://github.com/ccam80/cubie/commit/96ba1cb3d1fb6bb2bb0e6786ddb0e8dabd8512ae))
* buggy regex removed from pyproject ([bc5606a](https://github.com/ccam80/cubie/commit/bc5606aa67ce84994a5009f0b26dbd5d8e45e603))
* ignore generated python files ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))
* implement four-byte padding to reduce shared memory conflicts ([944066e](https://github.com/ccam80/cubie/commit/944066ee2ced3561b9607cbfcbfb48988f8256b9))
* jacobian product codegen now does a simple dead-code removal sweep at the expression level ([5a46559](https://github.com/ccam80/cubie/commit/5a46559f273a8e4408fbc777ef25c55dbfc06be8))
* metric function compilation now deferred until fn requested. ([fc42de7](https://github.com/ccam80/cubie/commit/fc42de7d438efb7968b4f4663016f9aae7cd82aa))
* sympy piecewise printing patched in subclass ([ca20f84](https://github.com/ccam80/cubie/commit/ca20f84061dda44c44f0ccac451d1aaa128ca4bc))
* SystemValues now interpreting sympy symbols correctly ([0549bea](https://github.com/ccam80/cubie/commit/0549beac411c21dd8a54167f517d8846a8500458))
* SystemValues, BaseODE now have comprehensible repros ([0549bea](https://github.com/ccam80/cubie/commit/0549beac411c21dd8a54167f517d8846a8500458))


### Documentation

* batchsolving module now has all docstrings in numpydocs-friendly format ([8e6e8cf](https://github.com/ccam80/cubie/commit/8e6e8cf415b86017137cd00421e14de58168e019))
* conf.py path reverted for Sphinx build ([5383de6](https://github.com/ccam80/cubie/commit/5383de6e551efde3ecf965efc50210608f136f12))
* docs updated and thinned to match structure ([9df6978](https://github.com/ccam80/cubie/commit/9df69786697ccf35f29f528a828a057a67d302ab))
* first-pass narrative docs added ([14577c1](https://github.com/ccam80/cubie/commit/14577c1e237791f2997ab53ffc842b5325763766))
* get sphinx-build working again and ReadTheDocs themed ([6df9a82](https://github.com/ccam80/cubie/commit/6df9a82fefa6307a9ef981ebe70ee2bc0c72161f))
* insert google verification tag, cross-link repo and docs ([0d61cd2](https://github.com/ccam80/cubie/commit/0d61cd2f93d1e79d2dc823a9830387d863cf8abb))
* integrators section docstrings brought in line with numpydocs format ([8b22790](https://github.com/ccam80/cubie/commit/8b22790c912b50f0681b7fb6d7eda925e87be64b))
* memory section docstrings brought into numpy format ([d495551](https://github.com/ccam80/cubie/commit/d495551cce6f0a150da76b6000f87ce512e4345a))
* output_functions section docstrings brought into numpy format ([4e7b2c0](https://github.com/ccam80/cubie/commit/4e7b2c06d1ec8b0ee873d97e0e30e0f4b42a849c))
* pypi version badge added ([8b22790](https://github.com/ccam80/cubie/commit/8b22790c912b50f0681b7fb6d7eda925e87be64b))
* readme now has code coverage badge ([8e6e8cf](https://github.com/ccam80/cubie/commit/8e6e8cf415b86017137cd00421e14de58168e019))
* odesystems section docstrings brought into numpy format ([9df6978](https://github.com/ccam80/cubie/commit/9df69786697ccf35f29f528a828a057a67d302ab))


### Miscellaneous Chores

* re-release 0.0.3 ([359abe3](https://github.com/ccam80/cubie/commit/359abe33cb49f47f6b4dcaacf3c630d82d95c69c))
* release 0.0.3 ([684ef91](https://github.com/ccam80/cubie/commit/684ef91d144178828b2ec5fe7bf3addd58b625a9))

## [0.0.2](https://github.com/ccam80/cubie/compare/v0.0.1...v0.0.2) (2025-08-18)


### Features

* BaseArrayManager class added to unify approach to allocating/deallocating device arrays through the memory manager ([416e363](https://github.com/ccam80/cubie/commit/416e3632085eaac507e596c22bafb34c22f107a2))
* BatchConfigurator now accepts extra user input types to match usage ([a345679](https://github.com/ccam80/cubie/commit/a3456797cbd37eb47a3d75b10b9d870fe6b22203))
* BatchInputArrays and BatchOutputArrays now subclass BaseArrayManager ([808695e](https://github.com/ccam80/cubie/commit/808695ed21eee0a777178babb2f181f6fc85afef))
* Memory Manager extended to queue and process requests from multiple objects ([67e66bf](https://github.com/ccam80/cubie/commit/67e66bfe82fc02258afcbb94d342621505632074))
* MemoryManager implemented ([3387142](https://github.com/ccam80/cubie/commit/3387142c7925c3ee76e60175bb1cddd0a1b87ce7))
* Solver interface now set up for real people to use (I think) ([2b05c1e](https://github.com/ccam80/cubie/commit/2b05c1eb1a590f7fcd99f4aa0d33171ea4f5de37))
* UserArrays now handles delivery from device arrays to end user for inspection ([9d93a7a](https://github.com/ccam80/cubie/commit/9d93a7a560500519b9cd500b70d082610fc9efc4))


### Bug Fixes

* BatchSolverKernel now uses blocksize to calculate dynamic shared memory correctly, and reduces blocksize if it's &gt; limit ([5d8263f](https://github.com/ccam80/cubie/commit/5d8263fd9aae11f6006f53bd349c44d169341405))
* bug in previosu commit: BatchSolverKernel now uses blocksize to calculate dynamic shared memory correctly, and reduces blocksize if it's &gt; limit ([afda7cb](https://github.com/ccam80/cubie/commit/afda7cbe7d72e2d3a0e0134dca1be49ca81dbe08))
* fix circular imports introduced in 636b5e3 ([2865160](https://github.com/ccam80/cubie/commit/2865160cc31fb7589cc75148dfbeaad88d78af37))
* force odd shared memory size per run to minimise clashes. ([fc12c76](https://github.com/ccam80/cubie/commit/fc12c768dbc7bd27bbd90b469dae8258c09149d9))
* forward declarations no longer cause circular imports ([e3841ff](https://github.com/ccam80/cubie/commit/e3841ffabdd745e1a5892417b19b7524a8d65f0a)), closes [#73](https://github.com/ccam80/cubie/issues/73)
* Newly-initialised memory manager no longer breaks in CUDA sim ([db99862](https://github.com/ccam80/cubie/commit/db99862da766224b1f1588fd15f41cb524dc40bc))
* output config flags now treated as derived quantities instead of attributes ([9d93a7a](https://github.com/ccam80/cubie/commit/9d93a7a560500519b9cd500b70d082610fc9efc4))
* pyproject.toml now points at correct license and readme files for building. ([f7bd7f0](https://github.com/ccam80/cubie/commit/f7bd7f00079c4e804cd92e9bfdcf18bd2089c42f))
* pyproject.toml now points at correct license file (but for real this time) ([93591ed](https://github.com/ccam80/cubie/commit/93591ed9c3c4465ff812b0402ec58c993d7cd63c))
* SolverKernel now solves and summarises accurately. ([879da10](https://github.com/ccam80/cubie/commit/879da1033c7e5318ea2633213d55501f9cb186f5))
* SolverKernel tests made CUDA-simulator-friendly ([ad702bb](https://github.com/ccam80/cubie/commit/ad702bb3a37f95e831723fc270978a5876316829))
* UserArrays now SolveResult, and works with array managers to produce a sensible output ([994cd7c](https://github.com/ccam80/cubie/commit/994cd7ccc6bcd53c1e02b73f7397d5fe0c4e1d65))
* UserArrays.as_numpy now returns copies rather than mappedarrays ([01af17d](https://github.com/ccam80/cubie/commit/01af17d493ceb74ac9b05468b929a7f6290cdd19))


### Documentation

* Docs don't mention CuMC anymore, and we shall never speak of it again ([fff2e00](https://github.com/ccam80/cubie/commit/fff2e0088168ac8106c50ff3d479cc49ccad4864))
* Docs updated to reflect cubie refactor ([205e748](https://github.com/ccam80/cubie/commit/205e7489360c458818904c4e2bd0860639d54acf))
* Properties which expose lower-level attributes are now docstringed as such ([a97d368](https://github.com/ccam80/cubie/commit/a97d368500ae618ae35ca0cce40ce54ebc98380b))


### Miscellaneous Chores

* release 0.0.2 ([60b3cd1](https://github.com/ccam80/cubie/commit/60b3cd1887078b2bbcb92ff0160bfc1222fddbd6))

## 0.0.1 (2025-08-01)


### Features

* BatchConfigurator implemented and passing tests. ([0bc00f2](https://github.com/ccam80/smc/commit/0bc00f22422238ac8fcfe68fc07691de07757945))
* BatchConfigurator implemented and passing tests. ([f4859e4](https://github.com/ccam80/smc/commit/f4859e448893f6a5738297cb9bca2a1987248fe9))
* BatchConfigurator implemented and passing tests. ([edf60fa](https://github.com/ccam80/smc/commit/edf60faf40febc0b652a7f18d8c79f7c72a8fe0a))
* first attempt at a batch configurator ([4d723ed](https://github.com/ccam80/smc/commit/4d723edb192159b97d999bd2e011d1c65d108b05))
* Initial dev version release. Not all features documented in changelog; some commit messages poorly named. Expect better changelogs in subsequent releases ([1a7691e](https://github.com/ccam80/smc/commit/1a7691e31352cd1e6ff66538c96caa223e2f364d))


### Bug Fixes

* Add a nozeros toggle to array sizes ([1300f3f](https://github.com/ccam80/smc/commit/1300f3f7285062bcda5ab7278ed0ab3cd2daa9de))
* **ci:** Fix release-please label permissions ([fa5d77f](https://github.com/ccam80/smc/commit/fa5d77f2292555475ec6e638cf11de0ccd90e45e))
* complete GPU tests on release tag ([f622a03](https://github.com/ccam80/smc/commit/f622a03e0445c6ae9088153c7a0811190eb81089))
* complete GPU tests on release tag ([f622a03](https://github.com/ccam80/smc/commit/f622a03e0445c6ae9088153c7a0811190eb81089))
* correct import error in ODEData.py ([cdd3144](https://github.com/ccam80/smc/commit/cdd3144e6a715284d6fc76f6670351f3f726607b))
* Corrected ancient typo in SystemValues that made all parameter setting invalid, confirmed test coverage ([c05f532](https://github.com/ccam80/smc/commit/c05f5326f4187c95c8c64cd0de68afa9ff193d96))
* fix circular dependency, improve arraysizes interface ([98ecf54](https://github.com/ccam80/smc/commit/98ecf54d0751491634d3fad39aa3a5331cdcf28f))
* **git:** close issue [#41](https://github.com/ccam80/smc/issues/41) ([da83993](https://github.com/ccam80/smc/commit/da83993e0997194fa3c569ab451868c53fe5a3c4))
* **git:** remove local junk from repo [#41](https://github.com/ccam80/smc/issues/41) ([a3cb122](https://github.com/ccam80/smc/commit/a3cb1224cb61a2da412b544512dcf5a5f9acf116))
* Implemented adapters for array size and allocation classes ([7a4da6a](https://github.com/ccam80/smc/commit/7a4da6a4f6a3c586e2024c29c5a924c4328c006a))
* Improve adapters and access to output_sizes objects through higher objects ([649e8cf](https://github.com/ccam80/smc/commit/649e8cf8e94e10dce28e67bcad303e46cb98efc0))
* Output array indices now gated by boolean flags to avoid memory access errors ([4f5628c](https://github.com/ccam80/smc/commit/4f5628ce3fad9e57c6a03bcde65a032bc7b27428))
* **OutputHandling:** switched indexing order in 2d arrays to match intended striding. ([86a1d2f](https://github.com/ccam80/smc/commit/86a1d2f5f7622b51f302c56e330b51a7293c542f))
* Plumbing now works between lower-level modules ([d79cfdb](https://github.com/ccam80/smc/commit/d79cfdb0817f6c1e8aa9977c2a6a5af6854e4b7a))
* remove bug introduced in time-saving ([ce96d23](https://github.com/ccam80/smc/commit/ce96d236b6cf5b8c7d87f3a1a4cc24a50cfa2ad4))
* remove doto [#19](https://github.com/ccam80/smc/issues/19) ([d3457bd](https://github.com/ccam80/smc/commit/d3457bd23d17f030840ee690601151e289ccf152))
* Remove todos to close issues ([f2e7638](https://github.com/ccam80/smc/commit/f2e76384e407c1d7ac8c2a36e80916b9c79bd3b3))
* Set buffer height methods in output_functions to properties to follow the convention for the rest of the module ([3a5c387](https://github.com/ccam80/smc/commit/3a5c387e1414b56b2cfb0d4e1fd0e19ba82769bd))
* SingleIntegratorRun and children now re-validate timing after an update ([0baad64](https://github.com/ccam80/smc/commit/0baad6433925af427a9f3b60cee4df71121c7e4d))
* Swat rename bug in summary_metrics testing ([cffd94d](https://github.com/ccam80/smc/commit/cffd94df0da154d8d85b11026b9fb53a59826bb8))


### Miscellaneous Chores

* release 0.0.1 ([73c85c5](https://github.com/ccam80/smc/commit/73c85c50c3c2991b2ae1e9237caf1aa2fd15316b))
