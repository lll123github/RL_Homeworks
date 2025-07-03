import matplotlib.pyplot as plt
def show_res(history):
    # Plot test results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(5, 1, 1)
    plt.plot(history[:, 0])
    plt.title('Cart Position')
    plt.ylabel('x (m)')
    
    plt.subplot(5, 1, 2)
    plt.plot(history[:, 1])
    plt.title('first Pole Angle')
    plt.ylabel('theta (rad)')

    plt.subplot(5, 1, 3)
    plt.plot(history[:, 2])
    plt.title('Second Pole Angle')
    plt.ylabel('theta (rad)')
    
    plt.subplot(5, 1, 4)
    plt.plot(history[:, 3])
    plt.title('Control Force')
    plt.ylabel('F (N)')
    plt.xlabel('Time step')

    plt.subplot(5, 1, 5)
    plt.plot(history[:, 4])
    plt.title('Reward')
    plt.ylabel('v')
    plt.xlabel('Time step')
    
    plt.tight_layout()
    plt.show()