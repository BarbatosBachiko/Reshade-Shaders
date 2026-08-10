/*----------------------------------------------|
| ::       BaBa Pipeline Configuration        :: |
|----------------------------------------------*/

#pragma once

// Public migration switch:
//   0 = BaBa Launcher (default)
//   1 = preserve the old per-effect pipeline
#if defined(BABA_USE_LEGACY_PIPELINE)
    #define _BABA_LEGACY_PIPELINE_WAS_DEFINED 1
#else
    #define _BABA_LEGACY_PIPELINE_WAS_DEFINED 0
#endif

#ifndef BABA_USE_LEGACY_PIPELINE
    #define BABA_USE_LEGACY_PIPELINE 0
#endif

// Presets that select an external provider are treated as legacy presets.
#if !defined(USE_BABA_LAUNCHER) && !_BABA_LEGACY_PIPELINE_WAS_DEFINED
    #if (defined(USE_MARTY_LAUNCHPAD_MOTION) && USE_MARTY_LAUNCHPAD_MOTION) || \
        (defined(USE_VORT_MOTION) && USE_VORT_MOTION) || \
        (defined(USE_LUMENITE_KERNEL_MOTION) && USE_LUMENITE_KERNEL_MOTION) || \
        (defined(USE_LUMENITE_QUANTMOTION) && USE_LUMENITE_QUANTMOTION)
        #undef BABA_USE_LEGACY_PIPELINE
        #define BABA_USE_LEGACY_PIPELINE 1
    #endif
#endif

// USE_BABA_LAUNCHER remains a source-compatibility alias for older presets.
#ifndef _BABA_USE_LAUNCHER
    #if defined(USE_BABA_LAUNCHER)
        #if USE_BABA_LAUNCHER
            #define _BABA_USE_LAUNCHER 1
        #else
            #define _BABA_USE_LAUNCHER 0
        #endif
    #elif BABA_USE_LEGACY_PIPELINE
        #define _BABA_USE_LAUNCHER 0
    #else
        #define _BABA_USE_LAUNCHER 1
    #endif
#endif

// Provider selector: 0 Launcher, 1 local BaBa Flow, 2 Marty, 3 Vort,
// 4 Lumenite Kernel, 5 Lumenite QuantMotion.
#ifndef _BABA_MOTION_PROVIDER
    #if _BABA_USE_LAUNCHER
        #define _BABA_MOTION_PROVIDER 0
    #elif defined(USE_MARTY_LAUNCHPAD_MOTION) && USE_MARTY_LAUNCHPAD_MOTION
        #define _BABA_MOTION_PROVIDER 2
    #elif defined(USE_VORT_MOTION) && USE_VORT_MOTION
        #define _BABA_MOTION_PROVIDER 3
    #elif defined(USE_LUMENITE_KERNEL_MOTION) && USE_LUMENITE_KERNEL_MOTION
        #define _BABA_MOTION_PROVIDER 4
    #elif defined(USE_LUMENITE_QUANTMOTION) && USE_LUMENITE_QUANTMOTION
        #define _BABA_MOTION_PROVIDER 5
    #else
        #define _BABA_MOTION_PROVIDER 1
    #endif
#endif
