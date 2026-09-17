# Release workflow

The manifests, `.github/workflows/release.yml`, and `scripts/release-matrix.py` are the source of truth.

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


## macOS architecture split

Starting with `v0.2.0-alpha.28`, publish `kcastle-desktop-macos-arm64.dmg` for Apple Silicon
and `kcastle-desktop-macos-x64.dmg` for Intel. Both the app and Velopack's `UpdateMac` are
single-architecture binaries, selected **before** Velopack signs the app and creates package
checksums. The release workflow checks both architectures and the final app signature before
uploading. The package ID remains `Kcastle`, and the channel remains `alpha`, `beta`, or `stable`;
only the feed directory changes to `<channel>/osx-arm64` or `<channel>/osx-x64`.

`v0.2.0-alpha.28` is also the **last Universal bridge release**. `MACOS_BRIDGE_TAG` in
`scripts/release-matrix.py` gates that extra build; do not advance it for later releases.
The normal release-version PR must use this tag for the first release containing the split.
The bridge's ARM64 slice already selects `alpha/osx-arm64`; its Intel slice selects
`alpha/osx-x64`. Existing alpha installations migrate as follows:

1. An old Universal app checks `alpha/osx-universal` and installs the Universal bridge.
2. After restart, the bridge checks the feed for its running architecture.
3. A **higher-version** release replaces it with the single-architecture app. Velopack will not
   install the same-version native package over the bridge; users can remain on the bridge until
   the next alpha release. Fresh installations can use either native DMG immediately.

Keep `alpha/osx-universal/releases.alpha.json` and every package it references indefinitely,
including the bridge's full `.nupkg`. The existing Universal DMG can remain too. Future releases
never upload to or delete from that prefix. Exclude these objects from R2 expiry/cleanup rules.
An old user returning months later must still be able to take the same two updates.
Do not rerun a pre-bridge release workflow against that prefix; it could overwrite the frozen
feed. Manual dispatch checks out the requested tag for both the matrix and app sources.
No Universal build is needed for subsequent alpha, beta, or stable releases.

### Migration verification

Run `python3 scripts/release-matrix.py --self-test` and
`cargo test -p kcastle-desktop --locked updater::tests`. The matrix check covers the one-time
bridge and later alpha/beta/stable releases; the updater tests cover native feed routing and
Velopack's version/Full-package selection through local feeds.

Before publishing the bridge, test on disposable app copies on Apple Silicon and Intel:

1. Serve local feeds/packages for an old version, the Universal bridge, and a higher native
   version, using the same `Kcastle` package ID and channel. Point the test builds to that server.
2. Update old -> bridge, restart, then bridge -> native and restart. Check the selected feed,
   retained sessions/settings, and `lipo -archs` for both `Contents/MacOS/kcastle` and `UpdateMac`.
3. With only same-version native packages available, the bridge must report no newer update.
4. Publish another native test version without changing the Universal feed; repeat from the
   old app to verify delayed users still reach the newest native version.

These native restart checks complement unit tests; local feed tests do not execute app replacement.

Local packaging/replacement smoke check: repackaged disposable copies of `alpha.27` as an
`alpha.28` Universal bridge and `alpha.29` native packages. Both main executables and updaters
matched their expected architectures; ZIP/nupkg contents, feed hashes, and signatures passed.
The real updater applied both hops with `--norestart`, on ARM64 and on x86_64 via Rosetta.
This checks replacement without launching the app; Intel hardware and UI restart remain release
acceptance checks. To repeat replacement on disposable bundles with absolute paths:

```sh
"$test_app/Contents/MacOS/UpdateMac" apply --norestart --silent \
  --rootDir "$test_app" --packageDir "$test_packages" --package "$next_full_package"
```
