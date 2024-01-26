package org.stocks.objects;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@NoArgsConstructor
@AllArgsConstructor
@ToString
@Data
public class StockObject {
    private String date;
    private String ticker;
    private double open;
    private double high;
    private double low;
    private double close;
    private double adjClose;
    private double volume;
}
