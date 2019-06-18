function reward()
    
    reward = data.health / 176 + 176 / (data.enemy_health + 1)
    
    if data.matches_won then
        return 150
    elseif data.enemy_matches_won then
        return -100
    elseif data.health > 88 then
        return reward
    else
        return 0
    end
    
end