function save_model(model::MichiBoostModel, filepath::AbstractString)
    open(io -> Serialization.serialize(io, model), filepath, "w")
    return nothing
end

function load_model(filepath::AbstractString)
    return open(io -> Serialization.deserialize(io), filepath, "r")
end
