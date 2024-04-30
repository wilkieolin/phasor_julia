using Pkg
Pkg.activate(".")

using PkgTemplates

t = Template(;
    user="wilkieolin",
    authors="Wilkie Olin-Ammentorp",
    julia=v"1.9",
    plugins=[
        ProjectFile(version=v"1.0.0-DEV"),
        SrcDir(file="PhasorNetworks.jl"),
        Tests(file="run_tests.jl"),
        License(; name="MIT"),
        Git(; manifest=true, ssh=true),
        GitHubActions(; x64=true),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
    ],
)

t("PhasorNetworks.jl")