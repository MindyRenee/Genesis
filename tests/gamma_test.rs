#[test]
fn binds_colored_regions() {
    // 8x8: two separated same-color fragments + one other-color block
    let mut g = vec![0u8; 64];
    for i in 0..3 {
        g[8 + i] = 5;
        g[5 * 8 + i] = 5;
    } // two fragments of color 5
    for i in 0..4 {
        g[3 * 8 + 4 + i] = 3;
    } // one block of color 3
    let objs = genesis::daemon::gamma::GammaField::bind(&g, 8, 8, 0, 40);
    assert!(!objs.is_empty(), "should bind objects");
    let c5: Vec<_> = objs.iter().filter(|o| o.color == 5).collect();
    let c3: Vec<_> = objs.iter().filter(|o| o.color == 3).collect();
    assert!(!c5.is_empty(), "color-5 fragments bind");
    assert_eq!(c3.len(), 1, "color-3 block binds once");
    // same-color fragments may merge via similarity coupling
    println!(
        "objects: {:?}",
        objs.iter()
            .map(|o| (o.color, o.size, o.centroid))
            .collect::<Vec<_>>()
    );
}
