'''
Credits: Alberto Castrignanò
            - albertocastrignano2@gmail.com
            - https://www.linkedin.com/in/albertocastrignano2/
            - https://github.com/AlbInitPolito
'''

methods = ["__init__", "reset", "render", "adapt_actions", "step", "save_performance", "check_performance", "_get_info",
"data_to_save", "load_data_to_save"]
attrs = ["num_continue", "num_discrete", "epochs", "iterations", "width","height",
"channels", "lr", "gamma", "output_dir"]
null_attrs = ["range_continue","dim_discrete","preload_model_weights", "preload_losses",
"preload_observations", "preload_performance", "preload_data_to_save"]

'''
NB: this method only checks for methods and attributes presence, not coherence in their values!
'''

def check_env(env):
    for m in methods:
        c = getattr(env, m, None)
        if c is None:
            raise ValueError("The environment needs a "+m+" method!")
        if not callable(c):
            raise ValueError("The attribute "+m+" must be a method!")
    for a in attrs:
        c = getattr(env, a, None)
        if c is None:
            raise ValueError("The environment needs a "+a+" attribute!")
        if a=="num_discrete":
            val = env.num_discrete
            if val>0:
                try:
                    getattr(env, "dim_discrete")
                    if env.dim_discrete==None:
                        raise ValueError("If num_discrete is not 0, then dim_discrete must be set!")
                except:
                    raise ValueError("If num_discrete is not 0, then dim_discrete must be set!")
        elif a=="num_continue":
            val = env.num_continue
            if val>0:
                try:
                    getattr(env, "range_continue")
                    if env.range_continue==None:
                        raise ValueError("If num_continue is not 0, then range_continue must be set!")
                except:
                    raise ValueError("If num_continue is not 0, then range_continue must be set!")
    try:
        for na in null_attrs:
            getattr(env, na)
    except:
        raise ValueError("The environment needs a "+na+" attribute!")