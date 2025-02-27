
def get_extra_config(quantization, bits):
    if quantization == "pact":
        return get_extra_config_pact(bits)
    elif quantization == "lsq":
        return get_extra_config_lsq(bits)
    elif quantization == "dorefa":
        return get_extra_config_dorefa(bits)

# =====================================>
# PACT
# =====================================>
def get_extra_config_pact(bits):
    
    extra_config = {
                'extra_qconfig_dict': {
                    'w_observer': 'MinMaxObserver',                             # custom weight observer
                    'a_observer': 'MinMaxObserver',                            #  custom activation observer
                    'w_fakequantize': 'DoReFaFakeQuantize',                # custom weight fake quantize function
                    'a_fakequantize': 'PACTFakeQuantize',                # custom activation fake quantize function
                    'w_qscheme': {
                        'bit': bits,                                             # custom bitwidth for weight,
                        'symmetry': True,                                     # custom whether quant is symmetric for weight,
                        'per_channel': False,                                 # custom whether quant is per-channel or per-tensor for weight,
                        'pot_scale': False,                                   # custom whether scale is power of two for weight.
                    },
                    'a_qscheme': {
                        'bit': bits,                                             # custom bitwidth for activation,
                        'symmetry': True,                                     # custom whether quant is symmetric for activation,
                        'per_channel': False,                                 # custom whether quant is per-channel or per-tensor for activation,
                        'pot_scale': False,                                   # custom whether scale is power of two for activation.
                    }
                }
            }
    
    return extra_config


# =====================================>
# LSQ
# =====================================>
def get_extra_config_lsq(bits):

    extra_config = {
                'extra_qconfig_dict': {
                    'w_observer': 'LSQObserver',                              # custom weight observer
                    'a_observer': 'LSQObserver',                              # custom activation observer
                    'w_fakequantize': 'LearnableFakeQuantize',                # custom weight fake quantize function
                    'a_fakequantize': 'LearnableFakeQuantize',                # custom activation fake quantize function
                    'w_qscheme': {
                        'bit': bits,                                             # custom bitwidth for weight,
                        'symmetry': True,                                     # custom whether quant is symmetric for weight,
                        'per_channel': False,                                 # custom whether quant is per-channel or per-tensor for weight,
                        'pot_scale': False,                                   # custom whether scale is power of two for weight.
                    },
                    'a_qscheme': {
                        'bit': bits,                                             # custom bitwidth for activation,
                        'symmetry': True,                                     # custom whether quant is symmetric for activation,
                        'per_channel': False,                                 # custom whether quant is per-channel or per-tensor for activation,
                        'pot_scale': False,                                   # custom whether scale is power of two for activation.
                    }
                }
            }
    
    return extra_config



# =====================================>
# DoReFa
# =====================================>
def get_extra_config_dorefa(bits):
    
    extra_config = {
                'extra_qconfig_dict': {
                    'w_observer': 'MinMaxObserver',                             # custom weight observer
                    'a_observer': 'MinMaxObserver',                            #  custom activation observer
                    'w_fakequantize': 'DoReFaFakeQuantize',                # custom weight fake quantize function
                    'a_fakequantize': 'DoReFaFakeQuantize',                # custom activation fake quantize function
                    'w_qscheme': {
                        'bit': bits,                                             # custom bitwidth for weight,
                        'symmetry': True,                                     # custom whether quant is symmetric for weight,
                        'per_channel': False,                                 # custom whether quant is per-channel or per-tensor for weight,
                        'pot_scale': False,                                   # custom whether scale is power of two for weight.
                    },
                    'a_qscheme': {
                        'bit': bits,                                             # custom bitwidth for activation,
                        'symmetry': True,                                     # custom whether quant is symmetric for activation,
                        'per_channel': False,                                 # custom whether quant is per-channel or per-tensor for activation,
                        'pot_scale': False,                                   # custom whether scale is power of two for activation.
                    }
                }
            }
    
    return extra_config