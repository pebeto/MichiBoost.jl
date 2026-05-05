function save_model(model::MichiBoostModel, filepath::AbstractString)
    JLD2.save_object(filepath, model)
    return nothing
end

function load_model(filepath::AbstractString)
    return JLD2.load_object(filepath)
end
