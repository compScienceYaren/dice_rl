from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()

def process_tensor_events(event_acc, variable_name, get_steps):

    # Get tensor events for the specified variable
    tensor_events = event_acc.Tensors(variable_name)
    
    # Process the tensor events
    data = []
    for tensor_event in tensor_events:
        w = tensor_event.wall_time
        s = tensor_event.step
        t = tf.make_ndarray(tensor_event.tensor_proto)
        data.append((w, s, t))

    # steps = [entry[1] for entry in data]
    # values = [entry[2].item() for entry in data]

    if get_steps:
        return [entry[1] for entry in data]
    else:
        return [entry[2].item() for entry in data]

def makePlot(file_name):

    event_acc = EventAccumulator(file_name, size_guidance={"scalars": 0})
    event_acc.Reload()

    steps = process_tensor_events(event_acc, 'nu_zero', get_steps=True)
    
    variables = ['nu_zero', 'lam', 'dual_step', 'constraint', 'nu_reg', 'zeta_reg', 'lagrangian', 'overall']
    variable_dict = {}
    # Iterate over each variable and process its tensor events
    for var in variables:
        variable_dict[var] = process_tensor_events(event_acc, var, get_steps=False)
    
    for var, values in variable_dict.items():
        print(f"{var}: {values}")
    
    print(f"{'steps'}: {steps}")

    # Plot the values
    plt.figure(figsize=(10, 6))

    for var, val in variable_dict.items():
        plt.plot(steps, val, label=var)

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Summary Scalars over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/plot.png')

# makePlot("/home/yaslan/dice_rl/tests/testdata/cartpole_tabularFalse_alpha0.0_seed0_numtraj10_maxtraj50/nlr0.0001_zlr0.0001_zerorFalse_preg0.0_dreg1.0_nreg1.0_pformFalse_fexp2.0_zposTrue_scaler1.0_shiftr0.0_transrNone/events.out.tfevents.1714925327.DESKTOP-UI5ASN5.30498.0.v2")
# makePlot("/home/yaslan/dice_rl/tests/testdata/cartpole_tabularFalse_alpha0.0_seed0_numtraj10_maxtraj50/nlr0.0001_zlr0.0001_zerorFalse_preg0.0_dreg1.0_nreg1.0_pformFalse_fexp2.0_zposTrue_scaler1.0_shiftr0.0_transrNone/events.out.tfevents.1715522651.DESKTOP-UI5ASN5.30020.0.v2")
makePlot("/home/yaslan/dice_rl/tests/testdata/cartpole_tabularFalse_alpha0.0_seed0_numtraj10_maxtraj50/nlr0.0001_zlr0.0001_zerorFalse_preg0.0_dreg1.0_nreg1.0_pformFalse_fexp2.0_zposTrue_scaler1.0_shiftr0.0_transrNone/events.out.tfevents.1715524590.DESKTOP-UI5ASN5.30980.0.v2")