import torch
from .base import ValHandler, PredictHandler, StepLogHandler

class EnergyValHandler(ValHandler):
    """
    In energy mode, val_log uses:
      vl_true = gscore_loss + atom_loss
    """
    def on_epoch_start(self, pl_module):
        pl_module._energy_val_buf = {
            "gscore": [],
            "gscore_pos": [],
            "atom": [],
            "vl_true": [],
        }

    def collect(self, pl_module, batch, y_hat, loss, losses_list, dataloader_idx=None):
        gscore_loss, gscore_pos_loss, cef_loss, atom_loss, *_ = losses_list

        g = torch.as_tensor(gscore_loss).detach()
        gp = torch.as_tensor(gscore_pos_loss).detach()
        a = torch.as_tensor(atom_loss).detach()

        vl_true = g + a

        buf = pl_module._energy_val_buf
        buf["gscore"].append(g)
        buf["gscore_pos"].append(gp)
        buf["atom"].append(a)
        buf["vl_true"].append(vl_true)

    def on_epoch_end(self, pl_module):
        buf = pl_module._energy_val_buf

        def _mean_stack(xs):
            if len(xs) == 0:
                return None
            t = torch.stack(xs)
            t_all = pl_module.all_gather(t).reshape(-1)
            return t_all.mean()

        vl_true = _mean_stack(buf["vl_true"])
        gscore  = _mean_stack(buf["gscore"])
        gpos    = _mean_stack(buf["gscore_pos"])
        atom    = _mean_stack(buf["atom"])

        if vl_true is not None:
            pl_module.log("vl_true", vl_true, on_epoch=True, sync_dist=True)
        if gscore is not None:
            pl_module.log("vl_gscore", gscore, on_epoch=True, sync_dist=True)
        if gpos is not None:
            pl_module.log("vl_gscore_pos", gpos, on_epoch=True, sync_dist=True)
        if atom is not None:
            pl_module.log("vl_atom", atom, on_epoch=True, sync_dist=True)

        buf["gscore"].clear()
        buf["gscore_pos"].clear()
        buf["atom"].clear()
        buf["vl_true"].clear()


class EnergyPredictHandler(PredictHandler):
    """Extract a usable output from y_hat."""
    def predict(self, pl_module, batch, y_hat):
        pass

class EnergyStepLogger(StepLogHandler):
    """
    losses_list: (gscore_loss, gscore_pos_loss, cef_loss, atom_loss, loss_crf, loss_aa, loss_kmkcat)
    """
    def train_step_log(self, pl_module, losses_list, batch_size):
        gscore_loss, gscore_pos_loss, _, atom_loss, *_ = losses_list

        pl_module.log("tl_gscore", gscore_loss, on_step=True, on_epoch=True,
                      prog_bar=True, sync_dist=True, batch_size=batch_size)
        pl_module.log("tl_gscore_pos", gscore_pos_loss, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=True, batch_size=batch_size)
        pl_module.log("tl_atom", atom_loss, on_step=True, on_epoch=True,
                      prog_bar=True, sync_dist=True, batch_size=batch_size)

    def val_step_log(self, pl_module, losses_list, batch_size):
        gscore_loss, gscore_pos_loss, _, atom_loss, *_ = losses_list
        vl_true = gscore_loss + atom_loss

        pl_module.log("vl_true", vl_true, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=batch_size)
        pl_module.log("vl_gscore", gscore_loss, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=batch_size)
        pl_module.log("vl_gscore_pos", gscore_pos_loss, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=batch_size)
        pl_module.log("vl_atom", atom_loss, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=batch_size)