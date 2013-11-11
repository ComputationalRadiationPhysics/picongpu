Right now, you have to apply the following patch to run on `taurus` due to a boost install bug:

```diff

diff --git a/src/picongpu/CMakeLists.txt b/src/picongpu/CMakeLists.txt
index 2770501..dfe7c65 100644
--- a/src/picongpu/CMakeLists.txt
+++ b/src/picongpu/CMakeLists.txt
@@ -57,6 +57,7 @@ FIND_PACKAGE(Threads REQUIRED)
 SET(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
 
 #Boost from system
+SET(Boost_USE_MULTITHREADED OFF)
 FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options regex)
 INCLUDE_DIRECTORIES(AFTER ${Boost_INCLUDE_DIRS})
 LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

```
