import cdtools
import os

@cdtools.tools.distributed.report_speed_test
def main():
    filename = os.environ.get('CDTOOLS_TESTING_DATA_PATH')
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

    # FancyPtycho is the workhorse model
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3, # Use 3 incoherently mixing probe modes
        oversampling=2, # Simulate the probe on a 2xlarger real-space array
        probe_support_radius=120, # Force the probe to 0 outside a radius of 120 pix
        propagation_distance=5e-3, # Propagate the initial probe guess by 5 mm
        units='mm', # Set the units for the live plots
        obj_view_crop=-50, # Expands the field of view in the object plot by 50 pix
    )

    device = 'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    # Remove or comment out plotting statements
    for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=40):
        if model.rank == 0 and model.epoch % 10:
            print(model.report())

    for loss in model.Adam_optimize(25, dataset,  lr=0.005, batch_size=40):
        if model.rank == 0 and model.epoch % 10:
            print(model.report())

    for loss in model.Adam_optimize(25, dataset,  lr=0.001, batch_size=40):
        if model.rank == 0 and model.epoch % 10:
            print(model.report())

    # This orthogonalizes the recovered probe modes
    model.tidy_probes()

    return model


if __name__ == '__main__':
    main()