#!/bin/bash


# config_b_field_pol_probe_ccbv_field
#     config_b_field_pol_probe_obr_field
#     config_b_field_pol_probe_obv_field
#     config_b_field_tor_probe_saddle_voltage
#     config_coil_current
#     config_solenoid_current
#     config_summary_ip
config_files=(
  config_equilibrium_psi
)

for filename in "${config_files[@]}"; do

    cmd=(
        python
        /home/ir-lore2/VAE_fairmast/src/vae_pipeline/vae_pipeline_visualization.py
        --config_file_path "/home/ir-lore2/VAE_fairmast/src/vae_pipeline/data/output/conv1d_vae_${filename}/${filename}.json"
        --config_task_file_path "src/vae_pipeline/configs/task_encoding_VAE.yaml"
    )

    echo ">>> ${cmd[*]}"
    "${cmd[@]}"

done

# python /home/ir-lore2/VAE_fairmast/src/vae_pipeline/vae_pipeline_visualization.py --config_file_path "/home/ir-lore2/VAE_fairmast/src/vae_pipeline/configs/config_coil_current_25ms.json" --config_task_file_path "src/vae_pipeline/configs/task_encoding_VAE.yaml"