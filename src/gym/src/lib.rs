enum Taxi {
    Full(Passenger),
    Empty
}

enum Tile {
    Wall,
    Street(Option<Taxi>),
    R(Option<Passenger>, Option<Taxi>),
    G(Option<Passenger>, Option<Taxi>),
    Y(Option<Passenger>, Option<Taxi>),
    B(Option<Passenger>, Option<Taxi>),
}

struct Passenger {
    desired_destination : Destination,
}

enum Destination {
    R,
    G,
    Y,
    B
}

enum Action {
    South,
    North,
    East,
    West,
    Pickup,
    Dropoff,
}

type Gym = [[Tile;6]; 6];

impl Gym {

}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
