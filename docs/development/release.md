# Release workflow

The manifests and `.github/workflows/release.yml` are the source of truth.

1. Create `release/<version>` from the default branch. Update the workspace version and exact
   internal dependency version together. Stable releases use a minor bump with patch `0`;
   prereleases use Cargo semver such as `0.2.0-alpha.1`.
2. Open a pull request and wait for every CI check to pass. The pull request author must merge it;
   agents and automation must not merge release pull requests.
3. After confirming the pull request was merged, publish a GitHub Release from the merged commit
   with tag `v<version>`. Mark alpha and beta releases as prereleases.

Publishing the GitHub Release triggers the workflow that publishes `kcastle-agent`, builds native
desktop installers, uploads GitHub Release assets, and updates the R2 feeds. The manual workflow
dispatch rebuilds desktop assets for an existing release tag without publishing the crate again.
