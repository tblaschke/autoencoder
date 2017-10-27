# coding=utf-8


from rdkit import Chem
from rdkit.Chem import Draw

class DynamicPlot(object):
    def __init__(self, n_steps, min, max, save_file=None):
        import os
        if os.environ.get('DISPLAY', '') == '':
            print('no display found. Using non-interactive Agg backend')
            import matplotlib
            matplotlib.use("Agg", warn=True, force=True)
            self.gui = False
        else:
            import matplotlib
            matplotlib.use("TKAgg", warn=True, force=True)
            self.gui = True

        from matplotlib import pyplot as plt
        #plt.ion()

        self.min = min
        self.max = max
        self.data = None
        self.updated = False
        self.file = save_file
        self.step = 0
        self.scores = [[], []]

        if self.gui:
            self.fig, (self.mol_ax, self.score_ax) = plt.subplots(2, 1,
                                                                  figsize=(14, 8),
                                                                  gridspec_kw={'height_ratios': [3.5, 1]})

            self.score_ax.set_xlim(0, n_steps)
            self.score_ax.set_ylim(min, max)
            self.score_ax.set_ylabel(r"$\mathrm{P_{active}}$")
            self.score_ax.set_xlabel(r"$\mathrm{Iteration\ Step}$")

            self.mol_ax.set_title(r"$\mathrm{Generated\ Molecules}$", y=0.97)
            self.mol_ax.axis("off")
            plt.tight_layout()

            plt.show(False)
            plt.draw()
            self.fig.canvas.draw()
            self.fig.canvas.update()
            self.fig.canvas.flush_events()




    def fastupdate(self, data, smiles=None):
        self.scores[0].append(data[0])
        if data[1] > self.max:
            self.scores[1].append(self.max)
        elif data[1] < self.min:
            self.scores[1].append(self.min)
        else:
            self.scores[1].append(data[1])

    def update(self, data, smis):
        if isinstance(data, float):
            data = [self.step, data]
            self.step += 1
        if self.file is not None:
            with open(self.file, "a") as fd:
                smis_str = ""
                for smi in smis:
                    smis_str = smis_str + " " + smi
                fd.write("{}\t{}\t{}\n".format(self.step, data[1], smis_str))
        data[1] = 1-data[1]
        self.scores[0].append(data[0])
        if data[1] > self.max:
            self.scores[1].append(self.max)
        elif data[1] < self.min:
            self.scores[1].append(self.min)
        else:
            self.scores[1].append(data[1])

        if self.gui:
            if not self.updated:
                self.data = self.score_ax.plot(self.scores[0],self.scores[1], "r-")[0] # Returns a tuple of line objects thus [0]
                self.updated = True

            self.data.set_data(self.scores)

            mols = []
            for smi in smis:
                mol = Chem.MolFromSmiles(smi)
                mols.append(mol)
                if len(mols) == 8:
                    break
            if len(mols) > 0:
                try:
                    mol_img = Draw.MolsToGridImage(mols, subImgSize=(400, 400), molsPerRow=4)
                    self.mol_ax.images = []
                    self.mol_ax.imshow(mol_img, interpolation="bicubic")
                except Exception:
                    pass

            self.fig.canvas.draw()
            self.fig.canvas.update()
            self.fig.canvas.flush_events()