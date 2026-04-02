cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SEEGLocalization)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE    "https://github.com/your-org/SEEGLocalization")
set(EXTENSION_CATEGORY    "Epilepsy")
set(EXTENSION_CONTRIBUTORS "G05 Team")
set(EXTENSION_DESCRIPTION  "Automatic SEEG electrode localization using 3D U-Net AI on MRI T1 + Post-Op CT.")
set(EXTENSION_ICONURL      "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS       "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SEEGLocalization)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
