CI runtime test for atomicPhysics
---------------------------------

AtomicInputData
===============

copy content of from `https://github.com/ComputationalRadiationPhysics/FLYonPICAtomicTestData`, for example by

```shell
git clone git@github.com:ComputationalRadiationPhysics/FLYonPICAtomicTestData.git
cp ./ChargeStates_Cu.txt ./atomicInputData/
cp ./AtomicStates_Cu.txt ./atomicInputData/
cp ./AutonomousTransitions_Cu.txt ./atomicInputData/
cp ./BoundBoundTransitions_Cu.txt ./atomicInputData/
cp ./BoundFreeTransitions_Cu.txt ./atomicInputData/
```

Validation
==========

Call `validation/validate.sh` from the root directory
