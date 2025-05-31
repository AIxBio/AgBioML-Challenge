# Basic EDA so we understand the data

import pandas as pd
import numpy as np

np.random.seed(42)


# Load from feather files
betas = pd.read_feather("betas_orig.arrow")
metadata = pd.read_feather("metadata_orig.arrow")


# Selected based on similar(ish) age-ranges and tissue types vs rest of studies
heldout_studies = [
    'GSE27317', 'GSE19711', 'E-GEOD-50660', 'E-GEOD-77955', 'E-GEOD-71955', 
    'E-GEOD-54399', 'E-GEOD-83334', 
    'E-GEOD-59457', 'GSE38608', 'E-GEOD-62867', 
    'GSE36642', 'GSE20242', 'E-GEOD-51954', 
    'E-MTAB-487', 'E-GEOD-56515', 'E-GEOD-73832', 
    'E-GEOD-59509', 'E-GEOD-72338', 'E-GEOD-61454', 
    'E-GEOD-53740', 'GSE57285', 'E-GEOD-58045',
    'E-GEOD-30870', 'E-GEOD-71245', 'GSE34257', 'TCGA_LUSC', 
    'E-GEOD-41169', 'E-GEOD-21232', 'E-GEOD-27044', 'E-GEOD-32867'
]

# Create heldout dataset
metadata_heldout = metadata[metadata.dataset.isin(heldout_studies)]
betas_heldout = betas.loc[metadata_heldout.index]
assert np.all(metadata_heldout.index == betas_heldout.index)
metadata_heldout.to_feather('eval/meta_heldout.arrow')
betas_heldout.to_feather('eval/betas_heldout.arrow')

# Create 50/50 public/private split AT THE STUDY LEVEL
# This ensures no study appears in both public and private sets
unique_studies = sorted(list(set(heldout_studies)))  # Sort for deterministic behavior
np.random.shuffle(unique_studies)

# Split studies 50/50
split_point = len(unique_studies) // 2
public_studies = unique_studies[:split_point]
private_studies = unique_studies[split_point:]

print(f"Public studies ({len(public_studies)}): {', '.join(sorted(public_studies))}")
print(f"Private studies ({len(private_studies)}): {', '.join(sorted(private_studies))}")

# Create public subset - all samples from public studies
metadata_heldout_public = metadata_heldout[metadata_heldout.dataset.isin(public_studies)]
betas_heldout_public = betas_heldout.loc[metadata_heldout_public.index]
assert np.all(metadata_heldout_public.index == betas_heldout_public.index)
metadata_heldout_public.to_feather('eval/meta_heldout_public.arrow')
betas_heldout_public.to_feather('agent/betas_heldout_public.arrow')

# Create private subset - all samples from private studies
metadata_heldout_private = metadata_heldout[metadata_heldout.dataset.isin(private_studies)]
betas_heldout_private = betas_heldout.loc[metadata_heldout_private.index]
assert np.all(metadata_heldout_private.index == betas_heldout_private.index)
metadata_heldout_private.to_feather('eval/meta_heldout_private.arrow')
betas_heldout_private.to_feather('agent/betas_heldout_private.arrow')

# Create agent dataset
metadata_agents = metadata[~metadata.dataset.isin(heldout_studies)]
betas_agents = betas.loc[metadata_agents.index]
assert np.all(metadata_agents.index == betas_agents.index)
metadata_agents.to_feather('agent/metadata.arrow')
betas_agents.to_feather('agent/betas.arrow')

# Print summary with study-level information
print(f"\nTotal samples: {len(metadata)}")
print(f"Agent samples: {len(metadata_agents)}")
print(f"Heldout samples: {len(metadata_heldout)}")
print(f"  - Public: {len(metadata_heldout_public)} samples from {len(public_studies)} studies")
print(f"  - Private: {len(metadata_heldout_private)} samples from {len(private_studies)} studies")

# Verify no study overlap between public and private
assert len(set(public_studies) & set(private_studies)) == 0, "Study overlap detected between public and private!"
print("\n✓ Verified: No study overlap between public and private sets")

# Get the breakdown of tissues and age ranges (tissue_type and age and gender columns)
print("Agent data:")
print(metadata_agents['tissue_type'].value_counts())
print(metadata_agents['age'].describe())
print(metadata_agents['gender'].value_counts())
print("Public heldout data:")
print(metadata_heldout_public['tissue_type'].value_counts())
print(metadata_heldout_public['age'].describe())
print(metadata_heldout_public['gender'].value_counts())
print("Private heldout data:")
print(metadata_heldout_private['tissue_type'].value_counts())
print(metadata_heldout_private['age'].describe())
print(metadata_heldout_private['gender'].value_counts())

print(f"\nFiles created:")
print("  Agent data: betas.arrow, metadata.arrow")
print("  Full heldout: betas_heldout.arrow, meta_heldout.arrow")
print("  Public heldout: betas_heldout_public.arrow, meta_heldout_public.arrow")
print("  Private heldout: betas_heldout_private.arrow, meta_heldout_private.arrow")




