{
  description = "Micro LLM / classifier training pipeline, provisioned as a podman container";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f system);

      # cpu: nixpkgs' torch, built and cached by Hydra.
      # cuda: torch-bin (upstream CUDA wheels, unfree, no local CUDA build); the
      # host supplies the driver through CDI (`--device nvidia.com/gpu=all`).
      variants = system: {
        cpu = import nixpkgs { inherit system; };
        cuda = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      };

      pythonFor =
        variant: pkgs:
        if variant == "cuda" then
          pkgs.python3.override { packageOverrides = _: super: { torch = super.torch-bin; }; }
        else
          pkgs.python3;

      src = nixpkgs.lib.fileset.toSource {
        root = ./experiments/micro_train;
        fileset = nixpkgs.lib.fileset.unions [
          ./experiments/micro_train/pyproject.toml
          ./experiments/micro_train/micro_train
          ./experiments/micro_train/tests
        ];
      };

      # The pipeline; its test suite runs in the build, so a package that
      # exists has passed it against exactly this torch.
      pipeline =
        python:
        python.pkgs.buildPythonApplication {
          pname = "micro-train";
          version = "0.1.0";
          pyproject = true;
          inherit src;
          build-system = [ python.pkgs.setuptools ];
          dependencies = with python.pkgs; [
            torch
            safetensors
          ];
          nativeCheckInputs = [ python.pkgs.pytestCheckHook ];
          pythonImportsCheck = [ "micro_train" ];
          meta.mainProgram = "micro-train";
        };

      # An OCI image streamed straight into `podman load`: no tarball in the
      # store, and the tag is the image's content hash, so a run always uses
      # the image this flake built.
      image =
        variant: pkgs: app:
        pkgs.dockerTools.streamLayeredImage {
          # Fully qualified: podman would otherwise file a bare name under docker.io/library.
          name = "localhost/micro-train";
          contents = [
            app
            pkgs.bashInteractive
            pkgs.coreutils
            pkgs.cacert
          ];
          extraCommands = "mkdir -p work tmp && chmod 1777 tmp";
          config = {
            Entrypoint = [ "${app}/bin/micro-train" ];
            Cmd = [ "--help" ];
            WorkingDir = "/work";
            Volumes."/work" = { };
            Env = [
              "HOME=/tmp"
              "PYTHONUNBUFFERED=1"
              "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            ]
            ++
              pkgs.lib.optional (variant == "cuda")
                # CDI mounts the host driver libraries at host paths; let torch find libcuda there.
                "LD_LIBRARY_PATH=/run/opengl-driver/lib:/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/lib";
            Labels."org.opencontainers.image.source" = "https://github.com/larsbx/llm-kernel-lab";
          };
        };

      # `nix run .#run -- lm --data corpus.txt --out runs/lm`: load the image if
      # podman lacks it, then run it with the current directory as /work.
      # Prefers the host's podman (rootless needs the host's setuid newuidmap).
      runner =
        variant: pkgs: img:
        pkgs.writeShellApplication {
          name = "micro-train-${variant}";
          text = ''
            podman="''${PODMAN:-$(command -v podman || echo ${pkgs.podman}/bin/podman)}"
            ref="${img.imageName}:${img.imageTag}"
            "$podman" image exists "$ref" || ${img} | "$podman" load
            exec "$podman" run --rm --userns=keep-id -v "$PWD:/work:Z" ${
              pkgs.lib.optionalString (variant == "cuda") "--device nvidia.com/gpu=all"
            } "$ref" "$@"
          '';
        };

      build =
        system:
        nixpkgs.lib.mapAttrs (
          variant: pkgs:
          let
            python = pythonFor variant pkgs;
            app = pipeline python;
            img = image variant pkgs app;
          in
          {
            inherit app img python;
            run = runner variant pkgs img;
            load = pkgs.writeShellApplication {
              name = "micro-train-load-${variant}";
              text = ''${img} | "''${PODMAN:-$(command -v podman || echo ${pkgs.podman}/bin/podman)}" load'';
            };
          }
        ) (variants system);
    in
    {
      packages = forAll (
        system:
        let
          b = build system;
        in
        {
          default = b.cpu.img;
          micro-train = b.cpu.app;
          image = b.cpu.img;
          image-cuda = b.cuda.img;
        }
      );

      apps = forAll (
        system:
        let
          b = build system;
          app = drv: {
            type = "app";
            program = nixpkgs.lib.getExe drv;
          };
        in
        {
          default = app b.cpu.run;
          run = app b.cpu.run;
          run-cuda = app b.cuda.run;
          load = app b.cpu.load;
          load-cuda = app b.cuda.load;
        }
      );

      devShells = forAll (
        system:
        let
          b = build system;
          pkgs = (variants system).cpu;
        in
        {
          default = pkgs.mkShell {
            packages = [
              (b.cpu.python.withPackages (
                ps: with ps; [
                  torch
                  safetensors
                  pytest
                ]
              ))
              pkgs.podman
            ];
            shellHook = ''export PYTHONPATH="$PWD/experiments/micro_train''${PYTHONPATH:+:$PYTHONPATH}"'';
          };
        }
      );

      checks = forAll (
        system:
        let
          b = build system;
        in
        {
          micro-train = b.cpu.app;
          image = b.cpu.img;
        }
      );

      formatter = forAll (system: (variants system).cpu.nixfmt);
    };
}
