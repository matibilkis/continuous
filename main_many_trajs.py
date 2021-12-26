from integrate import generate_traj

for ppp in tqdm([6000, 10000, 20000]):
    for k in range(10):
        generate_traj(ppp=ppp, periods = 5, itraj=0, path = "sanity/integration/{}/".format(ppp))
